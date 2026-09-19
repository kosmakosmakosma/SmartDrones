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
    for name in inspect.signature(dynamics_class).parameters.keys()
    if name != 'self'
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
    ckpt = torch.load(ckpt_path, map_location=args.device)
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
defender_exclusion_R = target_R + capture_R
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


def query_value(state_np, tau):
    state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    query = responder.query(state_tensor, torch.tensor([tau], dtype=torch.float32, device=device))
    return query.values[0].detach().cpu().item()


def query_controls(state_np, tau):
    """Attacker + defender controls and value at (state, time-to-go tau); real units, numpy in/out."""
    global nominal_attacker_controls, nominal_defender_controls
    state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    time_tensor = torch.tensor([tau], dtype=torch.float32, device=device)

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
print(f"Querying V with countdown tau=max({tMax} - t, 0)")
print(f"Initial V = {V0:.4f}")
print()

for i in range(n_steps):
    s = trajectory[i]
    tau = max(tMax - times[i], 0.0)

    # get attacker/defender controls (bang-bang or receding-horizon MPC)
    u, d, V_val = query_controls(s, tau)

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
    tau_new = max(tMax - times[i + 1], 0.0)
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
ax.set_title("Value function V(tMax - t, x(t))")
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
attacker_policy = 'MPC' if attacker_uses_mpc else 'NN bang-bang'
defender_policy = 'MPC' if defender_uses_mpc else 'NN bang-bang'
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

save_path = os.path.join(experiment_dir, f"{args.controller}_simulation_ckpt{args.checkpoint}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {save_path}")
plt.show()
