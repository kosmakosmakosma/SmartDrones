"""Benchmark joint MPC hyperparameters on CrazyflieInterception without a trained network.

Every rollout starts at time-to-go = horizon, so the score only depends on reach/avoid along the
trajectory and on boundary_fn at time-to-go 0 (which an 'exact' DeepReach model reproduces exactly).
The network is replaced by boundary_fn, which is only used for the bang-bang warm start.

Metrics per configuration (same initial states, seeds and reference opponents for every configuration):
  exploit       [best defender response to the returned attacker plan]
                - [best attacker response to the returned defender plan], each computed with a
                strong reference optimiser; 0 means neither player can improve, smaller is better
  attacker_gap  part of exploit due to the attacker plan (strong defender score - returned score)
  defender_gap  part of exploit due to the defender plan (returned score - strong attacker score)
  *_vs_base     paired per-state difference against the baseline configuration, with 95% CI;
                a CI that excludes 0 is a real difference
  def_kept      fraction of trajectories whose defender plan is still the unchanged warm start
  att_kept      same for the attacker
  seconds       wall time of the optimisation itself per seed (excluding reference best responses)

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
    """Per-state metrics for one configuration and one sampling seed.

    The reference best responses use seeds that depend only on `seed`, so every configuration
    is judged by the same strong opponents (common random numbers).
    """
    horizon_steps = round(horizon_s / dt)
    times = torch.full((states.shape[0],), horizon_steps * dt)
    query = responder.query(states, times)
    warm_u = query.controls[:, None].expand(-1, horizon_steps, -1).clone()
    warm_d = query.disturbances[:, None].expand(-1, horizon_steps, -1).clone()

    common = dict(dt=dt, horizon_steps=horizon_steps, num_samples=num_samples,
                  num_iterations=num_iterations, noise_fraction=noise_fraction,
                  hold_steps=hold_steps, chunk=chunk)
    generator = torch.Generator().manual_seed(1000 * seed)
    start = time.perf_counter()
    result = optimize_joint_sequences(
        states, times, warm_u, warm_d, responder, dynamics,
        make_config(dynamics.accel_max_a, **common), make_config(dynamics.accel_max_d, **common),
        generator=generator, use_network_terminal_value=True)
    seconds = time.perf_counter() - start

    strong = dict(dt=dt, horizon_steps=horizon_steps, num_samples=strong_samples,
                  num_iterations=strong_iterations, noise_fraction=0.25, hold_steps=5, chunk=chunk)
    best_attacker = best_response(states, times, result.controls, result.defender_controls,
                                  responder, dynamics, strong, 'attacker', 1000 * seed + 1)
    best_defender = best_response(states, times, result.controls, result.defender_controls,
                                  responder, dynamics, strong, 'defender', 1000 * seed + 2)
    return {
        'score': result.score,
        'exploit': best_defender - best_attacker,
        # how much a strong defender gains against the returned attacker plan (attacker weakness)
        'attacker_gap': best_defender - result.score,
        # how much a strong attacker gains against the returned defender plan (defender weakness)
        'defender_gap': result.score - best_attacker,
        'def_kept': (result.defender_controls == warm_d).all(-1).all(-1).float(),
        'att_kept': (result.controls == warm_u).all(-1).all(-1).float(),
    }, seconds


def mean_ci(values):
    """Mean and 95% confidence half-width of a 1-D tensor."""
    return values.mean().item(), 1.96 * values.std().item() / values.numel() ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='mpc_benchmark.csv')
    parser.add_argument('--num_states', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3, help='MPC sampling seeds per configuration')
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--defender_exclusion_R', type=float, default=0.15)
    parser.add_argument('--strong_samples', type=int, default=512)
    parser.add_argument('--strong_iterations', type=int, default=10)
    parser.add_argument('--check_samples', type=int, default=1024,
                        help='Stronger reference used once on the baseline to check the reference is strong enough')
    parser.add_argument('--check_iterations', type=int, default=20)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--state_seed', type=int, default=0)
    args = parser.parse_args()

    dynamics = CrazyflieInterception(0.25, 0.2, 5.0, 7.0, defender_exclusion_R=args.defender_exclusion_R)
    responder = BoundaryResponder(dynamics)
    torch.manual_seed(args.state_seed)
    states = sample_mpc_initial_states(dynamics, args.num_states, 'interception', attacker_velocity='inward')
    reference = dict(strong_samples=args.strong_samples, strong_iterations=args.strong_iterations)

    def evaluate(config, **overrides):
        """Average per-state metrics over seeds; returns (metrics, mean seconds per seed)."""
        options = dict(reference, **overrides)
        totals, seconds = None, 0.0
        for seed in range(args.seeds):
            metrics, elapsed = run_config(states, dynamics, responder, args.dt, chunk=args.chunk,
                                          seed=seed, **options, **config)
            totals = metrics if totals is None else {k: totals[k] + v for k, v in metrics.items()}
            seconds += elapsed
        return {k: v / args.seeds for k, v in totals.items()}, seconds / args.seeds

    base = dict(horizon_s=1.0, hold_steps=10, noise_fraction=0.25, num_samples=128, num_iterations=5)
    baseline, baseline_seconds = evaluate(base)

    check, _ = evaluate(base, strong_samples=args.check_samples, strong_iterations=args.check_iterations)
    shift, shift_ci = mean_ci(check['exploit'] - baseline['exploit'])
    print('reference check: exploit %.4f with %dx%d reference vs %.4f with %dx%d (paired diff %+.4f +- %.4f)' % (
        baseline['exploit'].mean(), args.strong_samples, args.strong_iterations,
        check['exploit'].mean(), args.check_samples, args.check_iterations, shift, shift_ci), flush=True)

    sweeps = {
        'baseline': [None],
        'horizon_s': [1.5, 2.0, 2.5, 3.0],
        'hold_steps': [1, 5, 25, 50],
        'noise_fraction': [0.1, 0.5, 1.0],
        'num_samples': [32, 64, 256, 512],
        'num_iterations': [1, 3, 10],
    }
    rows = []
    for name, values in sweeps.items():
        for value in values:
            config = dict(base) if value is None else dict(base, **{name: value})
            metrics, seconds = (baseline, baseline_seconds) if value is None else evaluate(config)
            row = dict(swept=name, value=value, **config, seconds=seconds)
            for key in ('exploit', 'attacker_gap', 'defender_gap', 'score'):
                row[key], row[key + '_ci'] = mean_ci(metrics[key])
            for key in ('exploit', 'attacker_gap', 'defender_gap'):
                # paired difference against the baseline on the same states and reference opponents
                row[key + '_vs_base'], row[key + '_vs_base_ci'] = mean_ci(metrics[key] - baseline[key])
            row['def_kept'] = metrics['def_kept'].mean().item()
            row['att_kept'] = metrics['att_kept'].mean().item()
            rows.append(row)
            print('%s=%s exploit=%.4f+-%.4f diff=%+.4f+-%.4f att_gap=%.4f def_gap=%.4f seconds=%.2f' % (
                name, value, row['exploit'], row['exploit_ci'], row['exploit_vs_base'],
                row['exploit_vs_base_ci'], row['attacker_gap'], row['defender_gap'], seconds), flush=True)

    with open(args.output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
