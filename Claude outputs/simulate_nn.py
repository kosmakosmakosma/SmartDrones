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


# ──────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--experiments_dir', type=str, default='./runs')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='-1 for model_final.pth, else epoch number')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dt', type=float, default=0.002, help='simulation timestep (s)')
parser.add_argument('--t_max_sim', type=float, default=3.0, help='max simulation time (s)')
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
# Neural network control law
# ──────────────────────────────────────────────────
def nn_control(state_np, tau):
    """
    Query the trained value function at (tau, state) and compute
    the optimal bang-bang control from the spatial gradients.

    Args:
        state_np: numpy array of shape (8,) — real-unit state
        tau: float — backward time to query (use tMax for receding horizon)

    Returns:
        u: numpy array (2,) — attacker control [u_ax, u_az]
        d: numpy array (2,) — defender control [d_dx, d_dz]
        V: float — value function at this state
        dvds: numpy array (8,) — spatial gradients of V
    """
    # build real-unit coordinate: [tau, state_0, ..., state_7]
    coord = torch.zeros(1, 1 + dynamics.state_dim)
    coord[0, 0] = tau
    coord[0, 1:] = torch.tensor(state_np, dtype=torch.float32)

    # convert to model input (normalized)
    model_input = dynamics.coord_to_input(coord).to(args.device)
    model_input.requires_grad_(True)

    # forward pass
    model_results = model({'coords': model_input})
    model_out = model_results['model_out'].squeeze(dim=-1)
    model_in = model_results['model_in']

    # compute value and gradients in real units
    V = dynamics.io_to_value(model_in, model_out)
    dv = dynamics.io_to_dv(model_in, model_out)

    # dv has shape (1, 9): [dvdt, dvds_0, ..., dvds_7]
    dvds = dv[0, 1:]  # spatial gradients only

    # optimal control from dynamics class
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(args.device)
    u_tensor = dynamics.optimal_control(state_tensor, dvds.unsqueeze(0))
    d_tensor = dynamics.optimal_disturbance(state_tensor, dvds.unsqueeze(0))

    # convert to numpy
    u = u_tensor[0].detach().cpu().numpy()
    d = d_tensor[0].detach().cpu().numpy()
    V_val = V[0].detach().cpu().item()
    dvds_np = dvds.detach().cpu().numpy()

    return u, d, V_val, dvds_np


# ──────────────────────────────────────────────────
# Physical parameters (read from dynamics)
# ──────────────────────────────────────────────────
target_R = dynamics.target_R
capture_R = dynamics.capture_R
accel_max_a = dynamics.accel_max_a
accel_max_d = dynamics.accel_max_d
Gz = dynamics.Gz
tMax = orig_opt.tMax  # training time horizon

print(f"\nPhysical parameters:")
print(f"  target_R = {target_R}, capture_R = {capture_R}")
print(f"  accel_max_a = {accel_max_a}, accel_max_d = {accel_max_d}")
print(f"  Gz = {Gz}")

# ──────────────────────────────────────────────────
# Dynamics (same as CrazyflieInterception.dsdt)
# ──────────────────────────────────────────────────
def step_dynamics(state, u, d, dt):
    """RK4 step for double integrator dynamics."""
    def f(s):
        dsdt = np.zeros(8)
        dsdt[0] = s[1]              # dpx_a/dt = vx_a
        dsdt[1] = u[0]              # dvx_a/dt = u_ax
        dsdt[2] = s[3]              # dpz_a/dt = vz_a
        dsdt[3] = u[1] + Gz         # dvz_a/dt = u_az + g
        dsdt[4] = s[5]              # dpx_d/dt = vx_d
        dsdt[5] = d[0]              # dvx_d/dt = d_dx
        dsdt[6] = s[7]              # dpz_d/dt = vz_d
        dsdt[7] = d[1] + Gz         # dvz_d/dt = d_dz + g
        return dsdt

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ──────────────────────────────────────────────────
# Initial conditions
# ──────────────────────────────────────────────────
# state = [px_a, vx_a, pz_a, vz_a, px_d, vx_d, pz_d, vz_d]
state0 = np.array([
    1.5,  0.0,    # attacker: x=1.5m, vx=0
    1.0,  0.0,    # attacker: z=1.0m, vz=0
   -1.0,  0.0,    # defender: x=-1.0m, vx=0
    0.5,  0.0,    # defender: z=0.5m, vz=0
])

# ──────────────────────────────────────────────────
# Simulate
# ──────────────────────────────────────────────────
dt = args.dt
n_steps = int(args.t_max_sim / dt)

trajectory = np.zeros((n_steps + 1, 8))
controls_a = np.zeros((n_steps, 2))
controls_d = np.zeros((n_steps, 2))
values = np.zeros(n_steps + 1)
times = np.zeros(n_steps + 1)

trajectory[0] = state0

# query V at initial state
with torch.no_grad():
    pass  # gradients needed inside nn_control
_, _, V0, _ = nn_control(state0, tMax)
values[0] = V0

outcome = "timeout"
outcome_time = args.t_max_sim

print(f"\nSimulating with dt={dt}, t_max={args.t_max_sim}")
print(f"Querying V at tau={tMax} (receding horizon)")
print(f"Initial V = {V0:.4f}")
print()

for i in range(n_steps):
    s = trajectory[i]

    # get optimal control from neural network
    u, d, V_val, dvds = nn_control(s, tMax)

    controls_a[i] = u
    controls_d[i] = d

    # step dynamics
    trajectory[i + 1] = step_dynamics(s, u, d, dt)
    times[i + 1] = times[i] + dt

    # query V at new state
    _, _, V_new, _ = nn_control(trajectory[i + 1], tMax)
    values[i + 1] = V_new

    # check termination
    pos_a = np.array([trajectory[i + 1, 0], trajectory[i + 1, 2]])
    pos_d = np.array([trajectory[i + 1, 4], trajectory[i + 1, 6]])

    dist_to_target = np.linalg.norm(pos_a)
    dist_to_defender = np.linalg.norm(pos_a - pos_d)

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
              f"d_target={dist_to_target:.3f}  d_capture={dist_to_defender:.3f}")

print(f"\nOutcome: {outcome} at t = {outcome_time:.4f}s")
print(f"Final attacker pos: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 2]:.3f})")
print(f"Final defender pos: ({trajectory[-1, 4]:.3f}, {trajectory[-1, 6]:.3f})")
print(f"Final V = {values[-1]:.4f}")


# ──────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Panel 1: 2D trajectory (x vs z) ---
ax = axes[0, 0]
ax.set_title("Trajectories in (x, z) plane")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
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

ax.plot(times, dist_target, "b-", lw=2, label="Attacker → origin")
ax.plot(times, dist_capture, "r-", lw=2, label="Attacker → defender")
ax.axhline(target_R, color="green", ls="--", alpha=0.7, label=f"Target radius")
ax.axhline(capture_R, color="orange", ls=":", alpha=0.7, label=f"Capture radius")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: value function ---
ax = axes[0, 2]
ax.set_title("Value function V(tMax, x(t))")
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
ax.set_title("Attacker control (NN bang-bang)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [m/s²]")

t_ctrl = times[:-1]
ax.step(t_ctrl, controls_a[:, 0], "b-", lw=1.5, label="u_ax", where="post")
ax.step(t_ctrl, controls_a[:, 1], "c-", lw=1.5, label="u_az", where="post")
ax.axhline(accel_max_a, color="gray", ls=":", alpha=0.5)
ax.axhline(-accel_max_a, color="gray", ls=":", alpha=0.5)
ax.set_ylim(-accel_max_a * 1.3, accel_max_a * 1.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 5: defender control ---
ax = axes[1, 1]
ax.set_title("Defender control (NN bang-bang)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [m/s²]")

ax.step(t_ctrl, controls_d[:, 0], "r-", lw=1.5, label="d_dx", where="post")
ax.step(t_ctrl, controls_d[:, 1], "m-", lw=1.5, label="d_dz", where="post")
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
ax.plot(times, trajectory[:, 3], "b--", lw=1.5, label="vz_a")
ax.plot(times, trajectory[:, 5], "r-", lw=1.5, label="vx_d")
ax.plot(times, trajectory[:, 7], "r--", lw=1.5, label="vz_d")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"NN-Controlled Simulation  |  ckpt={args.checkpoint}  |  "
    f"Outcome: {outcome} at t={outcome_time:.3f}s",
    fontsize=13, fontweight="bold")
fig.tight_layout()

save_path = os.path.join(experiment_dir, f"nn_simulation_ckpt{args.checkpoint}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {save_path}")
plt.show()
