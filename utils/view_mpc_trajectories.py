"""Export and interactively browse MPC attacker/defender state trajectories."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from matplotlib.widgets import Button


STATE_DIM = 8


def load_trajectories(checkpoint_path, num_initial_states, horizon_steps):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    replay = checkpoint.get("mpc_replay_buffer")
    if replay is None:
        raise ValueError("Checkpoint does not contain an MPC replay buffer")

    steps = horizon_steps + 1
    labels_per_run = num_initial_states * steps
    label_count = replay["states"].shape[0]
    complete_runs = label_count // labels_per_run
    discarded = label_count - complete_runs * labels_per_run
    if complete_runs == 0:
        raise ValueError(
            f"Need {labels_per_run} labels for one run, but only {label_count} exist"
        )

    start = discarded
    end = start + complete_runs * labels_per_run
    states = replay["states"][start:end].reshape(
        complete_runs, num_initial_states, steps, STATE_DIM
    ).numpy()
    times = replay["times"][start:end].reshape(
        complete_runs, num_initial_states, steps
    ).numpy()
    values = replay["values"][start:end].reshape(
        complete_runs, num_initial_states, steps
    ).numpy()
    return checkpoint, states, times, values, discarded


def first_event(trajectory, geometry):
    """(step, outcome) of the first terminal event, or (last step, 'timeout')."""
    attacker = trajectory[:, [0, 2]]
    defender = trajectory[:, [4, 6]]
    attacker_target = np.linalg.norm(attacker, axis=-1) <= geometry["target_R"]
    defender_breach = np.linalg.norm(defender, axis=-1) <= geometry["defender_exclusion_R"]
    captured = (np.linalg.norm(attacker - defender, axis=-1) <= geometry["capture_R"]) & ~defender_breach
    for step in range(trajectory.shape[0]):   # same precedence as the reach-avoid score: capture wins ties
        if captured[step]:
            return step, "attacker captured"
        if attacker_target[step]:
            return step, "attacker reached target"
        if defender_breach[step]:
            return step, "defender entered exclusion zone"
    return trajectory.shape[0] - 1, "timeout"


def save_export(output_path, checkpoint, states, times, values, discarded):
    np.savez_compressed(
        output_path,
        states=states,
        times=times,
        values=values,
        checkpoint_epoch=np.asarray(checkpoint.get("epoch", -1)),
        discarded_fifo_prefix_labels=np.asarray(discarded),
    )


def show_trajectory_browser(states, times, values, output_path, discarded, geometry):
    run_count, trajectory_count, step_count, _ = states.shape
    current_run = 0
    current_trajectory = 0

    figure, (path_axis, state_axis) = plt.subplots(1, 2, figsize=(13, 6))
    plt.subplots_adjust(bottom=0.18, wspace=0.28)

    def redraw():
        path_axis.clear()
        state_axis.clear()

        event_step, outcome = first_event(states[current_run, current_trajectory], geometry)
        trajectory = states[current_run, current_trajectory, :event_step + 1]
        trajectory_times = times[current_run, current_trajectory, :event_step + 1]
        trajectory_values = values[current_run, current_trajectory, :event_step + 1]

        path_axis.add_patch(Circle((0, 0), geometry["target_R"], fill=True, alpha=0.15, color="green",
                                   label="target (r=%.2f)" % geometry["target_R"]))
        path_axis.add_patch(Circle((0, 0), geometry["defender_exclusion_R"], fill=False, color="purple",
                                   linestyle="-.", linewidth=1.5,
                                   label="defender exclusion (r=%.2f)" % geometry["defender_exclusion_R"]))
        path_axis.add_patch(Circle((trajectory[-1, 4], trajectory[-1, 6]), geometry["capture_R"], fill=False,
                                   color="tab:orange", linestyle=":", linewidth=1.2,
                                   label="capture radius (r=%.2f)" % geometry["capture_R"]))

        path_axis.plot(
            trajectory[:, 0], trajectory[:, 2], "o-", markersize=3,
            linewidth=1.5, label="Attacker"
        )
        path_axis.plot(
            trajectory[:, 4], trajectory[:, 6], "o-", markersize=3,
            linewidth=1.5, label="Defender"
        )
        path_axis.scatter(trajectory[0, 0], trajectory[0, 2], marker="s", s=55)
        path_axis.scatter(trajectory[0, 4], trajectory[0, 6], marker="s", s=55)
        path_axis.set_title("Position paths")
        path_axis.set_xlabel("x position")
        path_axis.set_ylabel("y position")
        path_axis.axis("equal")
        path_axis.grid(alpha=0.3)
        path_axis.legend(loc="best", fontsize="small")

        elapsed = trajectory_times[0] - trajectory_times
        state_axis.plot(elapsed, trajectory[:, 0], label="attacker x")
        state_axis.plot(elapsed, trajectory[:, 2], label="attacker y")
        state_axis.plot(elapsed, trajectory[:, 4], "--", label="defender x")
        state_axis.plot(elapsed, trajectory[:, 6], "--", label="defender y")
        state_axis.set_title("Position over rollout")
        state_axis.set_xlabel("elapsed rollout time")
        state_axis.set_ylabel("position")
        state_axis.grid(alpha=0.3)
        state_axis.legend(loc="best", fontsize="small")

        score_start = trajectory_values[0]
        figure.suptitle(
            "MPC run %d/%d | trajectory %d/%d | initial suffix value %.5g\n%s after %.2fs"
            % (
                current_run + 1, run_count,
                current_trajectory + 1, trajectory_count,
                score_start, outcome, trajectory_times[0] - trajectory_times[-1],
            )
        )
        figure.canvas.draw_idle()

    def move(delta):
        nonlocal current_run, current_trajectory
        flat_index = current_run * trajectory_count + current_trajectory + delta
        flat_index %= run_count * trajectory_count
        current_run, current_trajectory = divmod(flat_index, trajectory_count)
        redraw()

    previous_axis = figure.add_axes((0.30, 0.04, 0.14, 0.07))
    next_axis = figure.add_axes((0.56, 0.04, 0.14, 0.07))
    previous_button = Button(previous_axis, "Previous")
    next_button = Button(next_axis, "Next")
    previous_button.on_clicked(lambda _event: move(-1))
    next_button.on_clicked(lambda _event: move(1))

    def on_key(event):
        if event.key in ("left", "p"):
            move(-1)
        elif event.key in ("right", "n", " "):
            move(1)

    figure.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print("Exported trajectory array shape:", states.shape)
    print("Discarded FIFO-prefix labels:", discarded)
    print("Saved data:", os.path.abspath(output_path))
    print("Use the buttons or Left/Right arrow keys to browse trajectories.")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint containing mpc_replay_buffer")
    parser.add_argument("--output", default="mpc_trajectories.npz")
    parser.add_argument("--num-initial-states", type=int, default=300)
    parser.add_argument("--horizon-steps", type=int, default=50)
    parser.add_argument("--target-R", type=float, default=0.25)
    parser.add_argument("--capture-R", type=float, default=0.2)
    parser.add_argument("--exclusion-R", type=float, default=None,
                        help="Defender exclusion radius (default: stored in the file, else target-R + capture-R)")
    args = parser.parse_args()

    checkpoint, states, times, values, discarded = load_trajectories(
        args.checkpoint, args.num_initial_states, args.horizon_steps
    )
    geometry = dict(checkpoint.get("scenario_geometry") or {})
    geometry.setdefault("target_R", args.target_R)
    geometry.setdefault("capture_R", args.capture_R)
    geometry.setdefault("defender_exclusion_R", geometry["target_R"] + geometry["capture_R"])
    if args.exclusion_R is not None:
        geometry["defender_exclusion_R"] = args.exclusion_R
    save_export(args.output, checkpoint, states, times, values, discarded)
    show_trajectory_browser(states, times, values, args.output, discarded, geometry)


if __name__ == "__main__":
    main()
