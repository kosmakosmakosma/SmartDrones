"""Export complete MPC state trajectories from a training checkpoint."""

import argparse
import os

import numpy as np
import torch


def export_trajectories(checkpoint_path, output_path, num_initial_states, horizon_steps):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    replay = checkpoint.get("mpc_replay_buffer")
    if replay is None:
        raise ValueError("Checkpoint does not contain an MPC replay buffer")

    trajectory_steps = horizon_steps + 1
    run_size = num_initial_states * trajectory_steps
    replay_size = replay["states"].shape[0]
    complete_runs = replay_size // run_size
    discarded_labels = replay_size - complete_runs * run_size
    if complete_runs == 0:
        raise ValueError(
            "Replay buffer does not contain one complete run: "
            f"{replay_size} labels available, {run_size} required"
        )

    start = discarded_labels
    end = start + complete_runs * run_size
    states = replay["states"][start:end].reshape(
        complete_runs, num_initial_states, trajectory_steps, -1
    )
    times = replay["times"][start:end].reshape(
        complete_runs, num_initial_states, trajectory_steps
    )
    values = replay["values"][start:end].reshape(
        complete_runs, num_initial_states, trajectory_steps
    )

    np.savez_compressed(
        output_path,
        states=states.numpy(),
        times=times.numpy(),
        values=values.numpy(),
        checkpoint_epoch=np.asarray(checkpoint.get("epoch", -1)),
        num_initial_states=np.asarray(num_initial_states),
        horizon_steps=np.asarray(horizon_steps),
        discarded_labels=np.asarray(discarded_labels),
    )
    print(f"checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"complete runs exported: {complete_runs}")
    print(f"trajectory shape: {tuple(states.shape)}")
    print(f"discarded FIFO-prefix labels: {discarded_labels}")
    print(f"saved: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Training checkpoint containing mpc_replay_buffer")
    parser.add_argument("output", help="Output .npz path")
    parser.add_argument("--num-initial-states", type=int, default=300)
    parser.add_argument("--horizon-steps", type=int, default=50)
    args = parser.parse_args()
    export_trajectories(
        args.checkpoint,
        args.output,
        args.num_initial_states,
        args.horizon_steps,
    )
