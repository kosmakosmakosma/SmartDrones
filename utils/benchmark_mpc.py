"""Benchmark joint MPC hyperparameters on CrazyflieInterception without a trained network.

Every rollout starts at time-to-go = horizon, so the score only depends on reach/avoid along the
trajectory and on boundary_fn at time-to-go 0 (which an 'exact' DeepReach model reproduces exactly).
The network is replaced by boundary_fn, which is only used for the bang-bang warm start.

Metrics per configuration (same initial states for every configuration):
  score        mean final joint score J_0 (attacker minimises, defender maximises)
  exploit      mean of [best defender response to the returned attacker plan]
               - [best attacker response to the returned defender plan], each computed with a
               strong reference optimiser; 0 means neither player can improve, smaller is better
  def_kept     fraction of trajectories whose defender plan is still the unchanged warm start
  att_kept     same for the attacker
  seconds      wall time of the optimisation itself (excluding the reference best responses)

Usage: python -m utils.benchmark_mpc --output results.csv
"""

import argparse
import csv
import time

import torch

from controllers.bang_bang import BangBangQuery
from controllers.mpc import MPCConfig, optimize_joint_sequences
from dynamics.dynamics import CrazyflieInterception
from utils.mpc_data import sample_mpc_initial_states


class BoundaryResponder:
    """Stand-in for NeuralBangBangController: V = boundary_fn, policy from its position gradient."""

    def __init__(self, dynamics):
        self.dynamics = dynamics

    def query(self, states, times):
        with torch.enable_grad():
            states = states.detach().clone().requires_grad_(True)
            values = self.dynamics.boundary_fn(states)
            gradient, = torch.autograd.grad(values.sum(), states)
        # V(t, x) ~ l(x + t v): the velocity gradient points along the position gradient
        dvds = torch.zeros_like(gradient)
        dvds[..., [1, 3, 5, 7]] = gradient[..., [0, 2, 4, 6]]
        return BangBangQuery(
            values=values.detach(), gradients=dvds,
            controls=self.dynamics.optimal_control(states, dvds).detach(),
            disturbances=self.dynamics.optimal_disturbance(states, dvds).detach())


def make_config(bound, dt, horizon_steps, num_samples, num_iterations, noise_fraction, hold_steps, chunk):
    return MPCConfig(
        dt=dt, horizon_steps=horizon_steps, num_samples=num_samples, num_iterations=num_iterations,
        noise_std=noise_fraction * bound,
        control_lower=torch.full((2,), -bound), control_upper=torch.full((2,), bound),
        candidate_chunk_size=chunk, control_hold_steps=hold_steps)


def best_response(states, times, controls, disturbances, responder, dynamics, strong, player, seed):
    """Strong single-player best response against the other player's fixed open-loop plan."""
    accel = {'attacker': dynamics.accel_max_a, 'defender': dynamics.accel_max_d}
    fixed = dict(strong, num_samples=1)   # one sample = the unchanged nominal plan
    attacker = make_config(accel['attacker'], **(strong if player == 'attacker' else fixed))
    defender = make_config(accel['defender'], **(strong if player == 'defender' else fixed))
    generator = torch.Generator().manual_seed(seed)
    return optimize_joint_sequences(
        states, times, controls, disturbances, responder, dynamics, attacker, defender,
        generator=generator, use_network_terminal_value=True).score


def run_config(states, dynamics, responder, dt, horizon_s, hold_steps, noise_fraction,
               num_samples, num_iterations, strong_samples, strong_iterations, chunk, seed):
    horizon_steps = round(horizon_s / dt)
    times = torch.full((states.shape[0],), horizon_steps * dt)
    query = responder.query(states, times)
    warm_u = query.controls[:, None].expand(-1, horizon_steps, -1).clone()
    warm_d = query.disturbances[:, None].expand(-1, horizon_steps, -1).clone()

    common = dict(dt=dt, horizon_steps=horizon_steps, num_samples=num_samples,
                  num_iterations=num_iterations, noise_fraction=noise_fraction,
                  hold_steps=hold_steps, chunk=chunk)
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    result = optimize_joint_sequences(
        states, times, warm_u, warm_d, responder, dynamics,
        make_config(dynamics.accel_max_a, **common), make_config(dynamics.accel_max_d, **common),
        generator=generator, use_network_terminal_value=True)
    seconds = time.perf_counter() - start

    strong = dict(dt=dt, horizon_steps=horizon_steps, num_samples=strong_samples,
                  num_iterations=strong_iterations, noise_fraction=0.25,
                  hold_steps=max(1, min(hold_steps, 5)), chunk=chunk)
    best_attacker = best_response(states, times, result.controls, result.defender_controls,
                                  responder, dynamics, strong, 'attacker', seed + 1)
    best_defender = best_response(states, times, result.controls, result.defender_controls,
                                  responder, dynamics, strong, 'defender', seed + 2)
    return {
        'score': result.score.mean().item(),
        'exploit': (best_defender - best_attacker).mean().item(),
        'def_kept': (result.defender_controls == warm_d).all(-1).all(-1).float().mean().item(),
        'att_kept': (result.controls == warm_u).all(-1).all(-1).float().mean().item(),
        'seconds': seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='mpc_benchmark.csv')
    parser.add_argument('--num_states', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--defender_exclusion_R', type=float, default=0.15)
    parser.add_argument('--strong_samples', type=int, default=512)
    parser.add_argument('--strong_iterations', type=int, default=10)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dynamics = CrazyflieInterception(0.25, 0.2, 5.0, 7.0, defender_exclusion_R=args.defender_exclusion_R)
    responder = BoundaryResponder(dynamics)
    torch.manual_seed(args.seed)
    states = sample_mpc_initial_states(dynamics, args.num_states, 'interception', attacker_velocity='inward')

    base = dict(horizon_s=1.0, hold_steps=10, noise_fraction=0.25, num_samples=128, num_iterations=5)
    sweeps = {
        'horizon_s': [1.0, 1.5, 2.0, 2.5, 3.0],
        'hold_steps': [1, 5, 10, 25, 50],
        'noise_fraction': [0.1, 0.25, 0.5, 1.0],
        'num_samples': [32, 64, 128, 256, 512],
        'num_iterations': [1, 3, 5, 10],
    }
    rows, seen = [], set()
    for name, values in sweeps.items():
        for value in values:
            config = dict(base, **{name: value})
            key = tuple(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            metrics = run_config(states, dynamics, responder, args.dt, chunk=args.chunk, seed=args.seed,
                                 strong_samples=args.strong_samples,
                                 strong_iterations=args.strong_iterations, **config)
            row = dict(swept=name, **config, **metrics)
            rows.append(row)
            print(' '.join('%s=%s' % (k, ('%.3f' % v) if isinstance(v, float) else v) for k, v in row.items()),
                  flush=True)

    with open(args.output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
