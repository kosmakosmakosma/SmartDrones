"""
Simulate 2D double-integrator attacker vs defender with bang-bang control.

Attacker: tries to reach the origin (target_R).
Defender: tries to intercept the attacker (capture_R).

Both use bang-bang control: full thrust toward/away from objective.
  - Attacker: thrust toward origin
  - Defender: thrust toward attacker

State: [px_a, vx_a, pz_a, vz_a, px_d, vx_d, pz_d, vz_d]
Controls: attacker u in [-accel_max_a, accel_max_a]^2
          defender d in [-accel_max_d, accel_max_d]^2
Dynamics: double integrator per axis with gravity on z
  dp/dt = v
  dv_x/dt = u_x  (or d_x for defender)
  dv_z/dt = u_z + Gz  (Gz = -9.8)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# ──────────────────────────────────────────────────
# Parameters (match your CrazyflieInterception class)
# ──────────────────────────────────────────────────
target_R = 0.25        # attacker wins if within this radius of origin
capture_R = 0.20       # defender wins if within this radius of attacker
accel_max_a = 5.0      # attacker max acceleration (m/s^2)
accel_max_d = 7.0      # defender max acceleration (m/s^2)
Gz = -9.8              # gravity on z-axis

dt = 0.001             # simulation timestep (s)
t_max = 3.0            # max simulation time (s)

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
# Bang-bang control laws
# ──────────────────────────────────────────────────
def attacker_control(state):
    """
    Bang-bang toward the origin.
    For each axis: if position > 0, we want to move negative, so thrust negative.
    Simple proportional-navigation-like heuristic:
      u_i = -accel_max * sign(p_i + damping * v_i)
    The damping term anticipates overshoot.
    """
    px_a, vx_a, pz_a, vz_a = state[0], state[1], state[2], state[3]

    # bang-bang toward origin with velocity damping
    damping = 0.3  # seconds of look-ahead
    u_ax = -accel_max_a * np.sign(px_a + damping * vx_a)
    u_az = -accel_max_a * np.sign((pz_a) + damping * (vz_a))
    # note: gravity is in the dynamics, not in the control decision

    return np.array([u_ax, u_az])


def defender_control(state):
    """
    Bang-bang toward the attacker (pursuit).
    For each axis: thrust in the direction of the attacker relative to defender.
      d_i = +accel_max_d * sign((p_attacker_i - p_defender_i) + damping * (v_attacker_i - v_defender_i))
    """
    px_a, vx_a, pz_a, vz_a = state[0], state[1], state[2], state[3]
    px_d, vx_d, pz_d, vz_d = state[4], state[5], state[6], state[7]

    damping = 0.3
    d_dx = accel_max_d * np.sign((px_a - px_d) + damping * (vx_a - vx_d))
    d_dz = accel_max_d * np.sign((pz_a - pz_d) + damping * (vz_a - vz_d))

    return np.array([d_dx, d_dz])


# ──────────────────────────────────────────────────
# Dynamics
# ──────────────────────────────────────────────────
def dynamics(state, u, d):
    """
    state: [px_a, vx_a, pz_a, vz_a, px_d, vx_d, pz_d, vz_d]
    u: [u_ax, u_az]  attacker acceleration
    d: [d_dx, d_dz]  defender acceleration
    """
    dsdt = np.zeros(8)
    # Attacker
    dsdt[0] = state[1]          # dpx_a/dt = vx_a
    dsdt[1] = u[0]              # dvx_a/dt = u_ax
    dsdt[2] = state[3]          # dpz_a/dt = vz_a
    dsdt[3] = u[1] + Gz         # dvz_a/dt = u_az + g
    # Defender
    dsdt[4] = state[5]          # dpx_d/dt = vx_d
    dsdt[5] = d[0]              # dvx_d/dt = d_dx
    dsdt[6] = state[7]          # dpz_d/dt = vz_d
    dsdt[7] = d[1] + Gz         # dvz_d/dt = d_dz + g
    return dsdt


# ──────────────────────────────────────────────────
# Simulate
# ──────────────────────────────────────────────────
n_steps = int(t_max / dt)
trajectory = np.zeros((n_steps + 1, 8))
controls_a = np.zeros((n_steps, 2))
controls_d = np.zeros((n_steps, 2))
times = np.zeros(n_steps + 1)

trajectory[0] = state0

outcome = "timeout"
outcome_time = t_max

for i in range(n_steps):
    s = trajectory[i]
    u = attacker_control(s)
    d = defender_control(s)
    controls_a[i] = u
    controls_d[i] = d

    # RK4 integration
    k1 = dynamics(s, u, d)
    k2 = dynamics(s + 0.5 * dt * k1, u, d)
    k3 = dynamics(s + 0.5 * dt * k2, u, d)
    k4 = dynamics(s + dt * k3, u, d)
    trajectory[i + 1] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    times[i + 1] = times[i] + dt

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
        times = times[:i + 2]
        break
    if dist_to_target <= target_R:
        outcome = "reached_target"
        outcome_time = times[i + 1]
        trajectory = trajectory[:i + 2]
        controls_a = controls_a[:i + 1]
        controls_d = controls_d[:i + 1]
        times = times[:i + 2]
        break

print(f"Outcome: {outcome} at t = {outcome_time:.4f}s")
print(f"Final attacker pos: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 2]:.3f})")
print(f"Final defender pos: ({trajectory[-1, 4]:.3f}, {trajectory[-1, 6]:.3f})")

# ──────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Panel 1: 2D trajectory (x vs z) ---
ax = axes[0, 0]
ax.set_title("Trajectories in (x, z) plane")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_aspect("equal")

# color trajectories by time
n_pts = len(trajectory)
t_norm = times / times[-1]  # normalize to [0, 1]

# attacker trajectory (blue gradient)
points_a = np.column_stack([trajectory[:, 0], trajectory[:, 2]])
segments_a = np.stack([points_a[:-1], points_a[1:]], axis=1)
lc_a = LineCollection(segments_a, cmap="Blues", linewidths=2)
lc_a.set_array(t_norm[:-1])
lc_a.set_clim(0, 1)
ax.add_collection(lc_a)

# defender trajectory (red gradient)
points_d = np.column_stack([trajectory[:, 4], trajectory[:, 6]])
segments_d = np.stack([points_d[:-1], points_d[1:]], axis=1)
lc_d = LineCollection(segments_d, cmap="Reds", linewidths=2)
lc_d.set_array(t_norm[:-1])
lc_d.set_clim(0, 1)
ax.add_collection(lc_d)

# markers for start and end
ax.plot(*points_a[0], "bs", markersize=10, label="Attacker start")
ax.plot(*points_a[-1], "b^", markersize=10, label="Attacker end")
ax.plot(*points_d[0], "rs", markersize=10, label="Defender start")
ax.plot(*points_d[-1], "r^", markersize=10, label="Defender end")

# target zone and capture zone at final positions
target_circle = Circle((0, 0), target_R, fill=False, color="green",
                        linewidth=2, linestyle="--", label=f"Target (r={target_R}m)")
ax.add_patch(target_circle)
capture_circle = Circle(points_a[-1], capture_R, fill=False, color="orange",
                        linewidth=2, linestyle=":", label=f"Capture zone (r={capture_R}m)")
ax.add_patch(capture_circle)

ax.autoscale_view()
ax.margins(0.1)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: distances over time ---
ax = axes[0, 1]
ax.set_title("Distances over time")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance [m]")

dist_target = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 2]**2)
dist_capture = np.sqrt((trajectory[:, 0] - trajectory[:, 4])**2 +
                        (trajectory[:, 2] - trajectory[:, 6])**2)

ax.plot(times, dist_target, "b-", linewidth=2, label="Attacker to origin")
ax.plot(times, dist_capture, "r-", linewidth=2, label="Attacker to defender")
ax.axhline(target_R, color="green", linestyle="--", alpha=0.7, label=f"Target radius ({target_R}m)")
ax.axhline(capture_R, color="orange", linestyle=":", alpha=0.7, label=f"Capture radius ({capture_R}m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: attacker control ---
ax = axes[1, 0]
ax.set_title("Attacker control (bang-bang)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [m/s²]")

t_ctrl = times[:-1]
ax.step(t_ctrl, controls_a[:, 0], "b-", linewidth=1.5, label="u_ax", where="post")
ax.step(t_ctrl, controls_a[:, 1], "c-", linewidth=1.5, label="u_az", where="post")
ax.axhline(accel_max_a, color="gray", linestyle=":", alpha=0.5)
ax.axhline(-accel_max_a, color="gray", linestyle=":", alpha=0.5)
ax.set_ylim(-accel_max_a * 1.3, accel_max_a * 1.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 4: defender control ---
ax = axes[1, 1]
ax.set_title("Defender control (bang-bang)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [m/s²]")

ax.step(t_ctrl, controls_d[:, 0], "r-", linewidth=1.5, label="d_dx", where="post")
ax.step(t_ctrl, controls_d[:, 1], "m-", linewidth=1.5, label="d_dz", where="post")
ax.axhline(accel_max_d, color="gray", linestyle=":", alpha=0.5)
ax.axhline(-accel_max_d, color="gray", linestyle=":", alpha=0.5)
ax.set_ylim(-accel_max_d * 1.3, accel_max_d * 1.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(f"Double Integrator: Attacker vs Defender  |  Outcome: {outcome} at t={outcome_time:.3f}s",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("simulation_result.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved to simulation_result.png")
