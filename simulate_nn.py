"""
Simulate 2D double-integrator attacker vs defender using the trained
DeepReach neural network value function for optimal control.

Run from the SmartDrones directory:
  python simulate_nn.py --experiment_name crazyflie_2d_v1 --checkpoint 5000

  --checkpoint -1    loads model_final.pth
  --checkpoint 5000  loads model_epoch_5000.pth (or whichever epoch)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import argparse
import os
import pickle
import inspect

from dynamics import dynamics as dynamics_module
from utils import modules
from controllers.bang_bang import NeuralBangBangController
from controllers.mpc import (
    MPCConfig,
    integrate_step,
    optimize_control_sequence,
    optimize_disturbance_sequence,
    optimize_joint_sequences,
    optimize_maxmin_sequences,
    shift_control_sequence,
)


# ──────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--experiments_dir', type=str, default='./runs')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='-1 for model_final.pth, else epoch number')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dt', type=float, default=0.02, help='simulation timestep (s)')
parser.add_argument('--t_max_sim', type=float, default=1.0, help='max simulation time (s)')
parser.add_argument('--fixed_time_to_go', type=float, default=None,
                    help='Query the network always at this time-to-go (e.g. 2.0 = tMax, the "unlimited time" value), '
                         'also inside the MPC rollouts; default: count down from tMax')
parser.add_argument(
    '--controller', type=str, default='bang_bang',
    choices=['bang_bang', 'mpc', 'attacker_mpc', 'defender_mpc', 'both_mpc'],
    help="'mpc' is an alias for attacker_mpc; 'both_mpc' uses alternating best responses",
)
parser.add_argument('--mpc_horizon_steps', type=int, default=50)
parser.add_argument('--mpc_control_hold_steps', type=int, default=10,
                    help='Number of rollout steps sharing each sampled control perturbation')
parser.add_argument('--mpc_num_samples', type=int, default=128)
parser.add_argument('--mpc_iterations', type=int, default=5)
parser.add_argument('--mpc_noise_fraction', type=float, default=0.25,
                    help='Gaussian std as a fraction of each optimized player max control')
parser.add_argument('--mpc_noise_std', type=float, default=None,
                    help='Absolute Gaussian std; overrides --mpc_noise_fraction')
parser.add_argument('--mpc_defender_noise_fraction', type=float, default=None,
                    help='Optional defender-specific noise fraction; defaults to --mpc_noise_fraction')
parser.add_argument('--mpc_defender_noise_std', type=float, default=None,
                    help='Absolute defender Gaussian std; overrides defender noise fraction')
parser.add_argument('--mpc_integrator', type=str, default='euler', choices=['euler', 'rk4'])
parser.add_argument('--mpc_seed', type=int, default=None)
parser.add_argument('--mpc_candidate_chunk_size', type=int, default=None)
parser.add_argument('--defender_policy', type=str, default='gradient', choices=['gradient', 'lookahead'],
                    help="network defender: 'gradient' = bang-bang from dV/d(defender velocity); 'lookahead' = try 9 "
                         "accelerations, predict --lookahead_time ahead (attacker keeps its velocity) and pick the one "
                         "with the highest V. Also used for the defender inside the attacker's MPC rollouts")
parser.add_argument('--lookahead_time', type=float, default=0.2, help='prediction time of the lookahead defender (s)')
parser.add_argument('--mpc_replan_every', type=int, default=1,
                    help='MPC player(s) re-optimise every N simulation steps and follow their plan in between; '
                         'the network player always acts on the actual state')
parser.add_argument('--mpc_opponent_model', type=str, default='network', choices=['network', 'mpc'],
                    help="'network': the MPC player simulates the network opponent inside its plans and ends them in the "
                         "network's value (it can exploit the network); 'mpc': the MPC player knows nothing about the "
                         "network: it plans the whole remaining game with the library's max-min MPC for both players "
                         "(pursuit, defender chase plans, the run's box rule, no network) and only its own moves are "
                         "executed")
parser.add_argument('--mpc_chase', type=lambda v: str(v).lower() in ('1', 'true', 'yes'), default=True,
                    help="with --mpc_opponent_model mpc: the planner's defender gets chase plans (as in the library)")
parser.add_argument('--safety_filter', type=lambda v: str(v).lower() in ('1', 'true', 'yes'), default=True,
                    help='network defender: reject accelerations after which (one step, then full braking) the '
                         'defender could no longer stop outside its keep-out zone (and, with the box rule, inside '
                         'the box); it then takes the best allowed one')
parser.add_argument('--start', type=float, nargs=8, default=None,
                    help='start state px_a vx_a py_a vy_a px_d vx_d py_d vy_d (default: the hard-coded state0)')
args = parser.parse_args()

experiment_dir = os.path.join(args.experiments_dir, args.experiment_name)

# ──────────────────────────────────────────────────
# Load trained model and dynamics
# ──────────────────────────────────────────────────
# load original training options
with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'rb') as f:
    orig_opt = pickle.load(f)

# instantiate dynamics class with original parameters
dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
dynamics_params = {
    name: getattr(orig_opt, name)
    for name, param in inspect.signature(dynamics_class).parameters.items()
    if name != 'self' and (hasattr(orig_opt, name) or param.default is inspect.Parameter.empty)
}
dynamics = dynamics_class(**dynamics_params)
dynamics.deepreach_model = orig_opt.deepreach_model

# instantiate model with original architecture
model = modules.SingleBVPNet(
    in_features=dynamics.input_dim,
    out_features=1,
    type=orig_opt.model,
    mode=orig_opt.model_mode,
    final_layer_factor=1.,
    hidden_features=orig_opt.num_nl,
    num_hidden_layers=orig_opt.num_hl,
)

# load checkpoint
checkpoints_dir = os.path.join(experiment_dir, 'training', 'checkpoints')
if args.checkpoint == -1:
    ckpt_path = os.path.join(checkpoints_dir, 'model_final.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    print(f"Loaded model_final.pth")
else:
    ckpt_path = os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)   # own training checkpoint (contains optimizer state)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded model_epoch_{args.checkpoint:04d}.pth")

model.to(args.device)
model.eval()

print(f"Dynamics: {orig_opt.dynamics_class}")
print(f"tMax (training horizon): {orig_opt.tMax}")
print(f"Architecture: {orig_opt.num_hl} hidden layers, {orig_opt.num_nl} neurons, {orig_opt.model} activation")

# ──────────────────────────────────────────────────
# Physical parameters (read from dynamics)
# ──────────────────────────────────────────────────
target_R = dynamics.target_R
capture_R = dynamics.capture_R
defender_exclusion_R = dynamics.defender_exclusion_R
accel_max_a = dynamics.accel_max_a
accel_max_d = dynamics.accel_max_d
tMax = orig_opt.tMax  # training time horizon

print(f"\nPhysical parameters:")
print(f"  target_R = {target_R}, capture_R = {capture_R}")
print(f"  defender_exclusion_R = {defender_exclusion_R}")
print(f"  accel_max_a = {accel_max_a}, accel_max_d = {accel_max_d}")

# ──────────────────────────────────────────────────
# Controllers (dynamics/integration live in `dynamics`; control policy lives here)
# ──────────────────────────────────────────────────
device = torch.device(args.device)
dt = args.dt
responder = NeuralBangBangController(model=model, dynamics=dynamics, device=device)


class FixedTimeResponder:
    """The network controller queried at one fixed time-to-go, whatever time it is asked about."""

    def __init__(self, inner, time_to_go):
        self.inner, self.time_to_go = inner, time_to_go

    def query(self, states, times):
        return self.inner.query(states, torch.full_like(times, self.time_to_go))

    def __getattr__(self, name):
        return getattr(self.inner, name)


if args.fixed_time_to_go is not None:
    responder = FixedTimeResponder(responder, args.fixed_time_to_go)


SAFETY_MARGIN = 0.005   # m kept clear of the keep-out circle and the box walls
safety_stats = {'steps': 0, 'changed': 0}


def braking_acceleration(velocity):
    """[..., 2] full braking: against the velocity, scaled so the larger axis is at accel_max_d."""
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    direction = velocity / speed.clamp_min(1e-9)
    scale = dynamics.accel_max_d / torch.amax(direction.abs(), dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.where(speed > 1e-6, -direction * scale, torch.zeros_like(velocity))


def defender_safe(states, accelerations):
    """[B, K] True where accelerating the defender with accelerations [B, K, 2] for one step and then braking
    fully keeps it out of its keep-out zone (and inside the box when the run's box rule is on)."""
    position = states[:, None, [4, 6]]
    velocity = states[:, None, [5, 7]].expand_as(accelerations)
    position = position + velocity * dt + 0.5 * accelerations * dt ** 2          # after one step
    velocity = velocity + accelerations * dt
    if dynamics.vel_max_d is not None:
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        velocity = velocity * torch.clamp(float(dynamics.vel_max_d) / speed.clamp_min(1e-9), max=1.0)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    direction = velocity / speed.clamp_min(1e-9)[..., None]
    deceleration = torch.linalg.vector_norm(braking_acceleration(velocity), dim=-1).clamp_min(1e-9)
    stop = speed ** 2 / (2 * deceleration)                                          # braking distance
    # the braking path is the straight segment position -> position + direction * stop
    along = torch.clamp(-(position * direction).sum(-1), min=0.0)
    along = torch.minimum(along, stop)
    closest = torch.linalg.vector_norm(position + direction * along[..., None], dim=-1)
    safe = closest > dynamics.defender_exclusion_R + SAFETY_MARGIN
    if getattr(dynamics, 'box_loses', False):
        end = position + direction * stop[..., None]
        limit = dynamics.pos_range - SAFETY_MARGIN
        safe = safe & (position.abs().amax(-1) < limit) & (end.abs().amax(-1) < limit)
    return safe


def choose_safe(states, chosen, candidates, scores):
    """Keep `chosen` [B, 2] where it is safe; elsewhere take the safe candidate [B, K, 2] with the highest score
    [B, K], or full braking if none is safe."""
    brake = braking_acceleration(states[:, [5, 7]])
    options = torch.cat((candidates, brake[:, None], chosen[:, None]), dim=1)
    safe = defender_safe(states, options)
    scores = torch.cat((scores, torch.full_like(scores[:, :1], -1e9), torch.full_like(scores[:, :1], -2e9)), dim=1)
    pick = torch.where(safe, scores, torch.full_like(scores, -float('inf'))).argmax(dim=1)
    pick = torch.where(safe[:, :-2].any(dim=1), pick, torch.full_like(pick, candidates.shape[1]))   # none: brake
    keep = safe[:, -1]
    safety_stats['steps'] += states.shape[0]
    safety_stats['changed'] += int((~keep).sum())
    return torch.where(keep[:, None], chosen, options[torch.arange(states.shape[0], device=states.device), pick])


class SafeGradientDefenderResponder:
    """Gradient (bang-bang) defender with the safety filter: an unsafe choice is replaced by the safe one among
    the 9 lookahead candidates that does best for V's linearisation (dV/d(defender velocity) . a)."""

    def __init__(self, inner):
        self.inner = inner
        a = dynamics.accel_max_d
        self.candidates = torch.tensor([[a, a], [a, -a], [-a, a], [-a, -a], [a, 0.0], [-a, 0.0], [0.0, a],
                                        [0.0, -a], [0.0, 0.0]], dtype=torch.float32, device=device)

    def query(self, states, times):
        query = self.inner.query(states, times)
        states = torch.as_tensor(states, dtype=torch.float32, device=device).reshape(-1, dynamics.state_dim)
        candidates = self.candidates[None].expand(states.shape[0], -1, -1)
        scores = (candidates * query.gradients[:, None, [5, 7]]).sum(-1)
        query.disturbances = choose_safe(states, query.disturbances, candidates, scores)
        return query

    def __getattr__(self, name):
        return getattr(self.inner, name)


class LookaheadDefenderResponder:
    """Network controller whose defender picks, among 9 accelerations (4 box corners, 4 axis directions, none),
    the one whose predicted state after `lookahead_time` has the highest V. The attacker keeps its current
    velocity during the prediction. Values only (no gradients), so V's position dependence is used directly."""

    def __init__(self, inner, lookahead_time, step_dt):
        self.inner = inner
        self.steps = max(int(round(lookahead_time / step_dt)), 1)
        self.step_dt = lookahead_time / self.steps
        a = dynamics.accel_max_d
        self.candidates = torch.tensor([[a, a], [a, -a], [-a, a], [-a, -a], [a, 0.0], [-a, 0.0], [0.0, a],
                                        [0.0, -a], [0.0, 0.0]], dtype=torch.float32, device=device)

    def _values(self, states, times):
        coords = dynamics.coord_to_input(torch.cat((times.unsqueeze(-1), states), dim=-1))
        with torch.no_grad():
            results = model({'coords': coords})
            return dynamics.io_to_value(results['model_in'], results['model_out'].squeeze(-1))

    def query(self, states, times):
        query = self.inner.query(states, times)
        states = torch.as_tensor(states, dtype=torch.float32, device=device).reshape(-1, dynamics.state_dim)
        times = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1).expand(states.shape[0])
        batch, count = states.shape[0], self.candidates.shape[0]
        predicted = states.repeat_interleave(count, dim=0)
        defender = self.candidates.repeat(batch, 1)
        attacker = torch.zeros_like(defender)              # attacker keeps its current velocity
        for _ in range(self.steps):
            predicted = integrate_step(dynamics, predicted, attacker, defender, self.step_dt, 'euler')
        if args.fixed_time_to_go is not None:
            value_times = torch.full((batch * count,), float(args.fixed_time_to_go), device=device)
        else:
            value_times = torch.clamp(times.repeat_interleave(count) - self.steps * self.step_dt, min=0.0)
        scores = self._values(predicted, value_times).view(batch, count)
        best = self.candidates[scores.argmax(dim=1)]
        if args.safety_filter:   # unsafe best: the best safe candidate by V, or full braking
            best = choose_safe(states, best, self.candidates[None].expand(batch, -1, -1), scores)
        query.disturbances = best
        return query

    def __getattr__(self, name):
        return getattr(self.inner, name)


if args.defender_policy == 'lookahead':
    responder = LookaheadDefenderResponder(responder, args.lookahead_time, dt)
elif args.safety_filter:
    responder = SafeGradientDefenderResponder(responder)

attacker_uses_mpc = args.controller in ('mpc', 'attacker_mpc', 'both_mpc')
defender_uses_mpc = args.controller in ('defender_mpc', 'both_mpc')
attacker_mpc_config = None
defender_mpc_config = None
mpc_generator = None
nominal_attacker_controls = None
nominal_defender_controls = None


def make_mpc_config(control_bound, action_dim, noise_fraction, noise_std):
    mpc_noise_std = noise_std if noise_std is not None else noise_fraction * control_bound
    config = MPCConfig(
        dt=dt,
        horizon_steps=args.mpc_horizon_steps,
        num_samples=args.mpc_num_samples,
        num_iterations=args.mpc_iterations,
        noise_std=mpc_noise_std,
        control_lower=torch.full((action_dim,), -control_bound),
        control_upper=torch.full((action_dim,), control_bound),
        integration_method=args.mpc_integrator,
        candidate_chunk_size=args.mpc_candidate_chunk_size,
        control_hold_steps=args.mpc_control_hold_steps,
    )
    return config, mpc_noise_std


if attacker_uses_mpc:
    attacker_mpc_config, noise_std = make_mpc_config(
        accel_max_a, dynamics.control_dim, args.mpc_noise_fraction, args.mpc_noise_std)
    print(f"  Attacker MPC noise std = {noise_std:.3f} "
          f"({noise_std / accel_max_a:.1%} of attacker max control)")
if defender_uses_mpc:
    defender_noise_fraction = (args.mpc_defender_noise_fraction
                               if args.mpc_defender_noise_fraction is not None
                               else args.mpc_noise_fraction)
    defender_noise_std = (args.mpc_defender_noise_std
                          if args.mpc_defender_noise_std is not None else args.mpc_noise_std)
    defender_mpc_config, noise_std = make_mpc_config(
        accel_max_d, dynamics.disturbance_dim,
        defender_noise_fraction, defender_noise_std)
    print(f"  Defender MPC noise std = {noise_std:.3f} "
          f"({noise_std / accel_max_d:.1%} of defender max control)")
if (attacker_uses_mpc or defender_uses_mpc) and args.mpc_seed is not None:
    mpc_generator = torch.Generator(device=device)
    mpc_generator.manual_seed(args.mpc_seed)


independent_mpc = args.mpc_opponent_model == 'mpc' and args.controller in ('mpc', 'attacker_mpc', 'defender_mpc')
independent_plans = {'u': None, 'd': None, 'start': 0}


def independent_mpc_actions(state_np, step):
    """Both players' actions from the MPC's own plan of the whole remaining game (max-min, no network)."""
    remaining = n_steps - step
    offset = step - independent_plans['start']
    if independent_plans['u'] is None or step % max(args.mpc_replan_every, 1) == 0:
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        if independent_plans['u'] is None:
            plan_u = torch.zeros(1, remaining, dynamics.control_dim, device=device)
            plan_d = torch.zeros(1, remaining, dynamics.disturbance_dim, device=device)
        else:   # receding horizon: drop the executed part, plans end with the game
            plan_u = independent_plans['u'][:, offset:offset + remaining]
            plan_d = independent_plans['d'][:, offset:offset + remaining]
        configs = []
        for bound, dim, player in ((accel_max_a, dynamics.control_dim, 'attacker'),
                                   (accel_max_d, dynamics.disturbance_dim, 'defender')):
            noise_std = args.mpc_noise_std if args.mpc_noise_std is not None else args.mpc_noise_fraction * bound
            keep_out = (dict(keep_out_dims=(4, 6), keep_out_radius=float(defender_exclusion_R))
                        if player == 'defender' else {})
            configs.append(MPCConfig(
                dt=dt, horizon_steps=remaining, num_samples=args.mpc_num_samples, num_iterations=args.mpc_iterations,
                noise_std=noise_std, control_lower=torch.full((dim,), -bound), control_upper=torch.full((dim,), bound),
                integration_method=args.mpc_integrator, candidate_chunk_size=args.mpc_candidate_chunk_size,
                control_hold_steps=args.mpc_control_hold_steps, pursuit=True,
                chase_top=4 if (player == 'defender' and args.mpc_chase) else 0, **keep_out))
        result = optimize_maxmin_sequences(
            state, torch.tensor([remaining * dt], device=device), plan_u, plan_d, None, dynamics,
            configs[0], configs[1], generator=mpc_generator)
        independent_plans.update(u=result.controls.detach(), d=result.defender_controls.detach(), start=step)
        offset = 0
    return (independent_plans['u'][0, offset].cpu().numpy(), independent_plans['d'][0, offset].cpu().numpy())


def query_value(state_np, tau):
    state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    query = responder.query(state_tensor, torch.tensor([tau], dtype=torch.float32, device=device))
    return query.values[0].detach().cpu().item()


def query_controls(state_np, tau, step=0):
    """Attacker + defender controls and value at (state, time-to-go tau); real units, numpy in/out."""
    global nominal_attacker_controls, nominal_defender_controls
    state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    time_tensor = torch.tensor([tau], dtype=torch.float32, device=device)

    if independent_mpc:   # the MPC player plans with its own model of the opponent, the network player reacts
        u_mpc, d_mpc = independent_mpc_actions(state_np, step)
        query = responder.query(state_tensor, time_tensor)
        u = u_mpc if attacker_uses_mpc else query.controls[0].detach().cpu().numpy()
        d = d_mpc if defender_uses_mpc else query.disturbances[0].detach().cpu().numpy()
        return u, d, query.values[0].detach().cpu().item()

    have_plans = ((not attacker_uses_mpc or nominal_attacker_controls is not None)
                  and (not defender_uses_mpc or nominal_defender_controls is not None))
    if args.controller != 'bang_bang' and have_plans and step % max(args.mpc_replan_every, 1) != 0:
        # between re-plans: MPC player(s) follow the current plan, the network player reacts to the actual state
        query = responder.query(state_tensor, time_tensor)
        u = (nominal_attacker_controls[0, 0] if attacker_uses_mpc else query.controls[0]).detach().cpu().numpy()
        d = (nominal_defender_controls[0, 0] if defender_uses_mpc else query.disturbances[0]).detach().cpu().numpy()
        if attacker_uses_mpc:
            nominal_attacker_controls = shift_control_sequence(nominal_attacker_controls)
        if defender_uses_mpc:
            nominal_defender_controls = shift_control_sequence(nominal_defender_controls)
        return u, d, query.values[0].detach().cpu().item()

    if args.controller == 'bang_bang':
        query = responder.query(state_tensor, time_tensor)
        u = query.controls[0].detach().cpu().numpy()
        d = query.disturbances[0].detach().cpu().numpy()
        V_val = query.values[0].detach().cpu().item()
        return u, d, V_val

    if ((attacker_uses_mpc and nominal_attacker_controls is None)
            or (defender_uses_mpc and nominal_defender_controls is None)):
        if attacker_uses_mpc and nominal_attacker_controls is None:
            nominal_attacker_controls = torch.zeros(
                1, args.mpc_horizon_steps, dynamics.control_dim,
                dtype=state_tensor.dtype, device=device,
            )
        if defender_uses_mpc and nominal_defender_controls is None:
            nominal_defender_controls = torch.zeros(
                1, args.mpc_horizon_steps, dynamics.disturbance_dim,
                dtype=state_tensor.dtype, device=device,
            )

    if attacker_uses_mpc and defender_uses_mpc:
        result = optimize_joint_sequences(
            state_tensor, time_tensor, nominal_attacker_controls, nominal_defender_controls,
            responder, dynamics, attacker_mpc_config, defender_mpc_config,
            generator=mpc_generator, use_network_terminal_value=True,
        )
    elif attacker_uses_mpc:
        result = optimize_control_sequence(
            state_tensor, time_tensor, nominal_attacker_controls,
            responder, dynamics, attacker_mpc_config,
            generator=mpc_generator, use_network_terminal_value=True,
        )
    else:
        result = optimize_disturbance_sequence(
            state_tensor, time_tensor, nominal_defender_controls,
            responder, dynamics, defender_mpc_config,
            generator=mpc_generator, use_network_terminal_value=True,
        )

    u = result.controls[0, 0].detach().cpu().numpy()
    d = result.defender_controls[0, 0].detach().cpu().numpy()
    V_val = result.network_values[0, 0].detach().cpu().item()
    if attacker_uses_mpc:
        nominal_attacker_controls = shift_control_sequence(result.controls.detach())
    if defender_uses_mpc:
        nominal_defender_controls = shift_control_sequence(result.defender_controls.detach())
    return u, d, V_val


# ──────────────────────────────────────────────────
# Initial conditions
# ──────────────────────────────────────────────────
# state = [px_a, vx_a, py_a, vy_a, px_d, vx_d, py_d, vy_d]
state0 = np.array([
    1.5,  0.0,    # attacker: x=1.5m, vx=0
    -0.2,  0.0,    # attacker: y=1.0m, vy=0
    0.0,  0.1,    # defender: x=-1.0m, vx=0
    0.7,  0.0,    # defender: y=0.5m, vy=0
])
if args.start is not None:
    state0 = np.array(args.start, dtype=float)

# ──────────────────────────────────────────────────
# Simulate
# ──────────────────────────────────────────────────
n_steps = int(args.t_max_sim / dt)

trajectory = np.zeros((n_steps + 1, 8))
controls_a = np.zeros((n_steps, 2))
controls_d = np.zeros((n_steps, 2))
values = np.zeros(n_steps + 1)
times = np.zeros(n_steps + 1)

trajectory[0] = state0

# query V at initial state
V0 = query_value(state0, tMax)
values[0] = V0

outcome = "timeout"
outcome_time = args.t_max_sim

print(f"\nSimulating with dt={dt}, t_max={args.t_max_sim}, controller={args.controller}")
print(f"Querying V at fixed time-to-go {args.fixed_time_to_go}" if args.fixed_time_to_go is not None
      else f"Querying V with countdown tau=max({tMax} - t, 0)")
print(f"Initial V = {V0:.4f}")
print()

for i in range(n_steps):
    s = trajectory[i]
    tau = (args.fixed_time_to_go if args.fixed_time_to_go is not None else max(tMax - times[i], 0.0))

    # get attacker/defender controls (bang-bang or receding-horizon MPC)
    u, d, V_val = query_controls(s, tau, step=i)

    controls_a[i] = u
    controls_d[i] = d

    # step dynamics (matches CrazyflieInterception.dsdt via RK4)
    state_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    control_tensor = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
    disturbance_tensor = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(0)
    next_state_tensor = integrate_step(dynamics, state_tensor, control_tensor, disturbance_tensor, dt, 'rk4')
    trajectory[i + 1] = next_state_tensor[0].detach().cpu().numpy()
    times[i + 1] = times[i] + dt

    # Querying the value must not trigger a second MPC optimization.
    tau_new = (args.fixed_time_to_go if args.fixed_time_to_go is not None else max(tMax - times[i + 1], 0.0))
    V_new = query_value(trajectory[i + 1], tau_new)
    values[i + 1] = V_new

    # check termination
    pos_a = np.array([trajectory[i + 1, 0], trajectory[i + 1, 2]])
    pos_d = np.array([trajectory[i + 1, 4], trajectory[i + 1, 6]])

    dist_to_target = np.linalg.norm(pos_a)
    dist_to_defender = np.linalg.norm(pos_a - pos_d)
    defender_dist_to_target = np.linalg.norm(pos_d)

    if defender_dist_to_target <= defender_exclusion_R:
        outcome = "defender_entered_target"
        outcome_time = times[i + 1]
        trajectory = trajectory[:i + 2]
        controls_a = controls_a[:i + 1]
        controls_d = controls_d[:i + 1]
        values = values[:i + 2]
        times = times[:i + 2]
        break
    if dist_to_defender <= capture_R:
        outcome = "captured"
        outcome_time = times[i + 1]
        trajectory = trajectory[:i + 2]
        controls_a = controls_a[:i + 1]
        controls_d = controls_d[:i + 1]
        values = values[:i + 2]
        times = times[:i + 2]
        break
    left = [name for name, pos in (('attacker', pos_a), ('defender', pos_d))
            if getattr(dynamics, 'box_loses', False) and np.max(np.abs(pos)) > dynamics.pos_range]
    if left:   # the run's box rule: leaving the box loses for the leaver
        outcome = "%s_left_box" % left[0]
        outcome_time = times[i + 1]
        trajectory = trajectory[:i + 2]
        controls_a = controls_a[:i + 1]
        controls_d = controls_d[:i + 1]
        values = values[:i + 2]
        times = times[:i + 2]
        break
    if dist_to_target <= target_R:
        outcome = "reached_target"
        outcome_time = times[i + 1]
        trajectory = trajectory[:i + 2]
        controls_a = controls_a[:i + 1]
        controls_d = controls_d[:i + 1]
        values = values[:i + 2]
        times = times[:i + 2]
        break

    # progress update every 100 steps
    if (i + 1) % 100 == 0:
        print(f"  t={times[i+1]:.3f}s  V={V_new:.4f}  "
              f"d_target={dist_to_target:.3f}  d_capture={dist_to_defender:.3f}  "
              f"d_defender_target={defender_dist_to_target:.3f}")

print(f"\nOutcome: {outcome} at t = {outcome_time:.4f}s")
print(f"Final attacker pos: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 2]:.3f})")
print(f"Final defender pos: ({trajectory[-1, 4]:.3f}, {trajectory[-1, 6]:.3f})")
print(f"Final V = {values[-1]:.4f}")
if args.safety_filter and safety_stats['steps']:
    print(f"Safety filter changed the network defender's action in {safety_stats['changed']} of "
          f"{safety_stats['steps']} queries (including queries inside MPC rollouts)")

state_mean = dynamics.state_mean.cpu().numpy()
state_var = dynamics.state_var.cpu().numpy()
outside_training_domain = np.logical_or(
    trajectory < state_mean - state_var,
    trajectory > state_mean + state_var,
)
if np.any(outside_training_domain):
    first_exit_step = np.flatnonzero(np.any(outside_training_domain, axis=1))[0]
    exited_dimensions = np.flatnonzero(outside_training_domain[first_exit_step])
    print(
        "WARNING: rollout left the sampled training domain at "
        f"t={times[first_exit_step]:.4f}s in state dimension(s) "
        f"{exited_dimensions.tolist()}; subsequent V and controls are extrapolation."
    )


# ──────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Panel 1: 2D trajectory (x vs y) ---
ax = axes[0, 0]
ax.set_title("Trajectories in (x, y) plane")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")

t_norm = times / times[-1]

points_a = np.column_stack([trajectory[:, 0], trajectory[:, 2]])
segments_a = np.stack([points_a[:-1], points_a[1:]], axis=1)
lc_a = LineCollection(segments_a, cmap="Blues", linewidths=2)
lc_a.set_array(t_norm[:-1])
ax.add_collection(lc_a)

points_d = np.column_stack([trajectory[:, 4], trajectory[:, 6]])
segments_d = np.stack([points_d[:-1], points_d[1:]], axis=1)
lc_d = LineCollection(segments_d, cmap="Reds", linewidths=2)
lc_d.set_array(t_norm[:-1])
ax.add_collection(lc_d)

ax.plot(*points_a[0], "bs", ms=10, label="Attacker start")
ax.plot(*points_a[-1], "b^", ms=10, label="Attacker end")
ax.plot(*points_d[0], "rs", ms=10, label="Defender start")
ax.plot(*points_d[-1], "r^", ms=10, label="Defender end")

ax.add_patch(Circle((0, 0), target_R, fill=False, color="green",
                     lw=2, ls="--", label=f"Target (r={target_R}m)"))
ax.add_patch(Circle((0, 0), defender_exclusion_R, fill=False, color="purple",
                     lw=2, ls="-.", label=f"Defender exclusion (r={defender_exclusion_R}m)"))
ax.add_patch(Circle(points_a[-1], capture_R, fill=False, color="orange",
                     lw=2, ls=":", label=f"Capture (r={capture_R}m)"))

ax.autoscale_view()
ax.margins(0.1)
ax.legend(loc="upper right", fontsize=7)
ax.grid(True, alpha=0.3)

# --- Panel 2: distances ---
ax = axes[0, 1]
ax.set_title("Distances over time")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance [m]")

dist_target = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 2]**2)
dist_capture = np.sqrt((trajectory[:, 0] - trajectory[:, 4])**2 +
                        (trajectory[:, 2] - trajectory[:, 6])**2)
dist_defender_target = np.sqrt(trajectory[:, 4]**2 + trajectory[:, 6]**2)

ax.plot(times, dist_target, "b-", lw=2, label="Attacker → origin")
ax.plot(times, dist_capture, "r-", lw=2, label="Attacker → defender")
ax.plot(times, dist_defender_target, color="purple", lw=2, label="Defender → origin")
ax.axhline(target_R, color="green", ls="--", alpha=0.7, label=f"Target radius")
ax.axhline(capture_R, color="orange", ls=":", alpha=0.7, label=f"Capture radius")
ax.axhline(defender_exclusion_R, color="purple", ls="-.", alpha=0.7, label="Defender exclusion")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: value function ---
ax = axes[0, 2]
ax.set_title("Value function V(%s, x(t))" % ('%.2f s' % args.fixed_time_to_go if args.fixed_time_to_go is not None
                                               else 'tMax - t'))
ax.set_xlabel("Time [s]")
ax.set_ylabel("V")

ax.plot(times, values, "k-", lw=2)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.fill_between(times, values, 0, where=(values <= 0), color="green", alpha=0.2, label="V ≤ 0 (attacker winning)")
ax.fill_between(times, values, 0, where=(values > 0), color="red", alpha=0.2, label="V > 0 (attacker losing)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 4: attacker control ---
ax = axes[1, 0]
mpc_name = 'MPC (own game model)' if independent_mpc else 'MPC'
network_name = ('NN lookahead' if args.defender_policy == 'lookahead' else 'NN bang-bang') + (
    ' + safety filter' if args.safety_filter else '')
attacker_policy = mpc_name if attacker_uses_mpc else 'NN bang-bang'
defender_policy = mpc_name if defender_uses_mpc else network_name
ax.set_title(f"Attacker control ({attacker_policy})")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [m/s²]")

t_ctrl = times[:-1]
ax.step(t_ctrl, controls_a[:, 0], "b-", lw=1.5, label="u_ax", where="post")
ax.step(t_ctrl, controls_a[:, 1], "c-", lw=1.5, label="u_ay", where="post")
ax.axhline(accel_max_a, color="gray", ls=":", alpha=0.5)
ax.axhline(-accel_max_a, color="gray", ls=":", alpha=0.5)
ax.set_ylim(-accel_max_a * 1.3, accel_max_a * 1.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 5: defender control ---
ax = axes[1, 1]
ax.set_title(f"Defender control ({defender_policy})")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [m/s²]")

ax.step(t_ctrl, controls_d[:, 0], "r-", lw=1.5, label="d_dx", where="post")
ax.step(t_ctrl, controls_d[:, 1], "m-", lw=1.5, label="d_dy", where="post")
ax.axhline(accel_max_d, color="gray", ls=":", alpha=0.5)
ax.axhline(-accel_max_d, color="gray", ls=":", alpha=0.5)
ax.set_ylim(-accel_max_d * 1.3, accel_max_d * 1.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 6: velocities ---
ax = axes[1, 2]
ax.set_title("Velocities")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Velocity [m/s]")

ax.plot(times, trajectory[:, 1], "b-", lw=1.5, label="vx_a")
ax.plot(times, trajectory[:, 3], "b--", lw=1.5, label="vy_a")
ax.plot(times, trajectory[:, 5], "r-", lw=1.5, label="vx_d")
ax.plot(times, trajectory[:, 7], "r--", lw=1.5, label="vy_d")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"{attacker_policy} Attacker vs {defender_policy} Defender  |  ckpt={args.checkpoint}  |  "
    f"Outcome: {outcome} at t={outcome_time:.3f}s",
    fontsize=13, fontweight="bold")
fig.tight_layout()

save_path = os.path.join(experiment_dir, f"{args.controller}_{args.mpc_opponent_model}-model_"
                         f"{args.defender_policy}{'-safe' if args.safety_filter else ''}_simulation_ckpt{args.checkpoint}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {save_path}")
plt.show()
