# MPC-Guided DeepReach Implementation Specification

.\env\Scripts\python.exe run_experiment.py `
  --mode train `
  --experiment_class DeepReach `
  --experiment_name crazyflie_2d_no_grav_test3 `
  --dynamics_class CrazyflieInterception `
  --minWith target `
  --target_R 0.25 `
  --capture_R 0.2 `
  --accel_max_a 5.0 `
  --accel_max_d 7.0 `
  --tMax 1.0 `
  --pretrain `
  --pretrain_iters 10000 `
  --counter_end 90000 `
  --num_epochs 200000 `
  --numpoints 45000 `
  --learned_boundary_fraction 0.5 `
  --geometric_boundary_fraction 0.2 `
  --learned_boundary_update_epochs 1000 `
  --learned_boundary_candidate_samples 100000 `
  --learned_boundary_keep_samples 10000 `
  --learned_boundary_buffer_size 100000 `
  --lr 2e-5 `
  --use_mpc_guidance `
  --mpc_start_epoch 100000 `
  --mpc_optimized_player joint `
  --mpc_initial_guess network `
  --mpc_state_distribution interception `
  --mpc_dt 0.02 `
  --mpc_horizon_steps 50 `
  --mpc_control_hold_steps 10 `
  --mpc_num_samples 128 `
  --mpc_iterations 5 `
  --mpc_noise_fraction 0.25 `
  --mpc_num_initial_states 300 `
  --mpc_refresh_epochs 1000 `
  --mpc_candidate_chunk_size 16 `
  --mpc_replay_capacity 200000 `
  --mpc_batch_size 1000 `
  --mpc_loss_weight 0.1 `
  --mpc_seed 1 `
  --use_wandb `
  --wandb_project SmartDrones `
  --wandb_entity kosmakosmakosma-tu-delft `
  --wandb_group 2d_drones_no_grav `
  --wandb_name test3

## 1. Objective

Refactor control logic out of `simulate_nn.py` and implement a modular sampling-based MPC system that:

1. Samples attacker control sequences around a nominal sequence.
2. Rolls out each attacker candidate.
3. Uses the neural value gradient to generate the defender's closed-loop bang-bang response.
4. Scores candidates using the correct BRAT reach-before-capture objective.
5. Selects and iteratively refines the best attacker sequence.
6. Supports receding-horizon simulation.
7. Supports batched MPC-label generation for neural-network training.

The existing BRAT PDE loss remains the primary self-supervised training loss. MPC supervision is an additional data loss, not a replacement.

## 2. Required File Structure

```text
controllers/
    __init__.py
    bang_bang.py       # Neural value/gradient queries and bang-bang policies
    mpc.py             # Sampling, integration, rollout, scoring, optimization

utils/
    mpc_data.py        # Replay buffer and training dataset for MPC labels

tests/
    test_bang_bang.py
    test_mpc.py
    test_mpc_data.py

simulate_nn.py         # Loading, problem setup, simulation, termination, plots
```

A partial `controllers/bang_bang.py` already exists and should be completed rather than duplicated.

## 3. Game and Sign Conventions

`CrazyflieInterception` has:

```python
loss_type = "brat_hjivi"
set_mode = "reach"
```

The attacker is the control player and minimizes the value. The defender is the disturbance player and maximizes the value.

Therefore:

- Lower trajectory scores are better for the attacker.
- MPC must select candidates with `argmin`, not `argmax`.
- `dynamics.optimal_control` produces the attacker's bang-bang action.
- `dynamics.optimal_disturbance` produces the defender's bang-bang action.

The intended MPC mode samples attacker control sequences while the defender follows the current neural feedback policy.

## 4. State Convention

The joint state has eight dimensions:

```text
[
    attacker_x,
    attacker_vx,
    attacker_y,
    attacker_vy,
    defender_x,
    defender_vx,
    defender_y,
    defender_vy,
]
```

The attacker control is:

```text
[attacker_ax, attacker_ay]
```

The defender disturbance is:

```text
[defender_ax, defender_ay]
```

Do not hard-code these state indices in generic MPC functions. Use the methods on `dynamics` wherever possible.

## 5. Tensor Conventions

All controller and MPC internals must use Torch tensors. NumPy conversion is allowed only at plotting or user-interface boundaries.

```text
B = initial-state batch size
N = sampled candidate sequences per initial state
H = number of rollout control steps
S = dynamics.state_dim
U = dynamics.control_dim
D = dynamics.disturbance_dim

initial_states:       [B, S]
initial_times:        [B]
nominal_controls:     [B, H, U]
sampled_controls:     [B, N, H, U]
state_trajectories:   [B, N, H+1, S]
defender_controls:    [B, N, H, D]
network_values:       [B, N, H+1]
trajectory_times:     [B, N, H+1]
trajectory_scores:    [B, N]
```

Public functions may accept an unbatched state `[S]` or control sequence `[H,U]`, but must normalize it to batched form internally. Return-shape behavior must be documented and consistent.

## 6. Time Convention

This repository uses time-to-go `tau`, not forward simulation time.

```text
tau = training_horizon - elapsed_simulation_time
tau_k = max(tau_initial - k * dt, 0)
```

The exact DeepReach model enforces its terminal condition at `tau = 0`. Simulation currently queries:

```python
tau = max(tMax - elapsed_time, 0.0)
```

All MPC neural queries and generated training labels must follow the same convention. Do not store elapsed forward time as the network coordinate.

A rollout must not integrate beyond its available time-to-go. For simulation, use:

```python
effective_horizon = min(config.horizon_steps, ceil(tau / dt))
```

For batched training with different time-to-go values, either group samples by effective horizon or maintain a valid-step mask and freeze each trajectory after its time-to-go reaches zero.

## 7. `controllers/bang_bang.py`

### 7.1 Query Result

Retain or implement:

```python
from dataclasses import dataclass
import torch


@dataclass
class BangBangQuery:
    values: torch.Tensor          # [B]
    gradients: torch.Tensor       # [B, S]
    controls: torch.Tensor        # [B, U], attacker controls
    disturbances: torch.Tensor    # [B, D], defender controls
```

### 7.2 Neural Controller

```python
class NeuralBangBangController:
    def __init__(self, model, dynamics, device): ...

    def query(
        self,
        states: torch.Tensor,     # [B,S] or [S], in real physical units
        times: torch.Tensor,      # [B] or scalar, time-to-go
    ) -> BangBangQuery: ...

    def __call__(self, states, times) -> BangBangQuery: ...
```

### 7.3 Query Algorithm

1. Convert states to `float32` on the configured device.
2. Convert a single state `[S]` to `[1,S]`.
3. Validate that the final state dimension equals `dynamics.state_dim`.
4. Convert times to `[B]`; broadcast a scalar or one-element tensor to the state batch.
5. Form real coordinates `[tau, *state]` with shape `[B,S+1]`.
6. Call `dynamics.coord_to_input(real_coordinates)`.
7. Run the model inside `torch.enable_grad()` because spatial derivatives are required.
8. Read `model_results["model_in"]` and `model_results["model_out"]`.
9. Call `dynamics.io_to_value(model_in, model_out)`.
10. Call `dynamics.io_to_dv(model_in, model_out)`.
11. Extract spatial gradients with `derivatives[..., 1:]`.
12. Call `dynamics.optimal_control(states, gradients)`.
13. Call `dynamics.optimal_disturbance(states, gradients)`.
14. Detach values, gradients, controls and disturbances before returning.

Do not wrap this query in `torch.no_grad()`. Calculating the policy requires the value gradient with respect to the coordinates.

During label generation, temporarily disable gradients on model parameters to reduce memory usage, but retain coordinate gradients:

```python
was_training = model.training
requires_grad = [parameter.requires_grad for parameter in model.parameters()]
model.eval()
model.requires_grad_(False)
try:
    # MPC generation; controller internally uses torch.enable_grad().
    ...
finally:
    for parameter, required in zip(model.parameters(), requires_grad):
        parameter.requires_grad_(required)
    model.train(was_training)
```

## 8. `controllers/mpc.py` Data Classes

```python
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MPCConfig:
    dt: float
    horizon_steps: int
    num_samples: int
    num_iterations: int
    noise_std: float
    control_lower: torch.Tensor
    control_upper: torch.Tensor
    integration_method: str = "euler"
    candidate_chunk_size: int | None = None


@dataclass
class MPCRollout:
    states: torch.Tensor
    attacker_controls: torch.Tensor
    defender_controls: torch.Tensor
    network_values: torch.Tensor
    times: torch.Tensor
    valid_steps: torch.Tensor | None = None


@dataclass
class MPCResult:
    controls: torch.Tensor
    states: torch.Tensor
    defender_controls: torch.Tensor
    network_values: torch.Tensor
    suffix_values: torch.Tensor
    score: torch.Tensor
    all_scores: torch.Tensor
```

### 8.1 Configuration Validation

Validate when optimization starts:

- `dt > 0`.
- `horizon_steps >= 1`.
- `num_samples >= 1`.
- `num_iterations >= 1`.
- `noise_std >= 0`.
- `integration_method` is `"euler"` or `"rk4"`.
- Bounds have shape `[U]` or are scalar-broadcastable to `[U]`.
- Every lower bound is strictly less than its upper bound.
- Nominal controls lie on the same device as the states.

For `CrazyflieInterception`, construct bounds from:

```python
lower = torch.full((dynamics.control_dim,), -dynamics.accel_max_a)
upper = torch.full((dynamics.control_dim,), dynamics.accel_max_a)
```

## 9. Control Sequence Sampling

Implement:

```python
def sample_control_sequences(
    nominal: torch.Tensor,             # [B,H,U]
    num_samples: int,
    noise_std: float | torch.Tensor,
    lower: torch.Tensor,               # [U]
    upper: torch.Tensor,               # [U]
    generator: torch.Generator | None = None,
) -> torch.Tensor:                     # [B,N,H,U]
    ...
```

Algorithm:

1. Validate nominal shape `[B,H,U]`.
2. Expand nominal over a new candidate dimension to `[B,N,H,U]`.
3. Generate independent Gaussian noise of the same shape.
4. Multiply noise by `noise_std`.
5. Add noise to the expanded nominal sequences.
6. Clamp each control dimension to `[lower, upper]`.
7. Set candidate zero exactly equal to the nominal sequence.
8. Return a contiguous tensor on the nominal device and dtype.

Candidate zero guarantees that an MPC refinement round cannot discard the current best sequence.

Optional later improvement: generate temporally correlated noise or optimize piecewise-constant control knots. Do not add this until the independent-Gaussian implementation is tested.

## 10. Dynamics Integration

Implement:

```python
def integrate_step(
    dynamics,
    states: torch.Tensor,          # [...,S]
    controls: torch.Tensor,        # [...,U]
    disturbances: torch.Tensor,    # [...,D]
    dt: float,
    method: str,
) -> torch.Tensor:                 # [...,S]
    ...
```

### 10.1 Euler

```python
next_states = states + dt * dynamics.dsdt(states, controls, disturbances)
```

Euler is the paper-compatible default and matches the repository's training rollout utilities.

### 10.2 RK4

For RK4, hold the selected attacker and defender actions constant through all four stages:

```python
k1 = dynamics.dsdt(states, controls, disturbances)
k2 = dynamics.dsdt(states + 0.5 * dt * k1, controls, disturbances)
k3 = dynamics.dsdt(states + 0.5 * dt * k2, controls, disturbances)
k4 = dynamics.dsdt(states + dt * k3, controls, disturbances)
next_states = states + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
```

After either method, call:

```python
next_states = dynamics.equivalent_wrapped_state(next_states)
```

Do not clamp physical states to the neural training domain. Clamping changes the physical system. Detect domain exits and optionally penalize or reject those candidates instead.

## 11. Batched Closed-Loop Rollout

Implement:

```python
def rollout_control_sequences(
    initial_states: torch.Tensor,       # [B,S]
    initial_times: torch.Tensor,        # [B]
    attacker_controls: torch.Tensor,    # [B,N,H,U]
    responder: NeuralBangBangController,
    dynamics,
    dt: float,
    integration_method: str = "euler",
) -> MPCRollout:
    ...
```

### 11.1 Algorithm

1. Validate all dimensions.
2. Expand `initial_states[:,None,:]` over candidates to `[B,N,S]`.
3. Allocate output tensors for states, controls, values and times.
4. Store the initial state at trajectory index zero.
5. For every step `k` from `0` through `H-1`:
   1. Compute `tau_k = clamp(initial_times - k*dt, min=0)`.
   2. Expand `tau_k` over candidates.
   3. Flatten current states from `[B,N,S]` to `[B*N,S]`.
   4. Flatten times from `[B,N]` to `[B*N]`.
   5. Query `responder.query(flat_states, flat_times)`.
   6. Reshape returned disturbances to `[B,N,D]`.
   7. Read sampled attacker controls `attacker_controls[:,:,k,:]`.
   8. Integrate one step with `integrate_step`.
   9. For samples whose time-to-go has expired, retain the previous state instead.
   10. Store state, disturbance, value and time.
   11. Detach the next state before the next neural query to prevent graph accumulation.
6. Query and store the value at the final state/time.
7. Return `MPCRollout`.

The defender action cannot be generated for the entire horizon before rollout. Its future state depends on its earlier actions, and each action depends on the NN gradient at the current joint state.

### 11.2 Convenience Wrapper

Implement the requested initial-state/control-sequence function:

```python
def rollout_trajectory(
    initial_state: torch.Tensor,             # [S]
    horizon_steps: int,
    attacker_control_sequence: torch.Tensor, # [H,U]
    initial_time: float | torch.Tensor,
    responder: NeuralBangBangController,
    dynamics,
    dt: float,
    integration_method: str = "euler",
) -> MPCRollout:
    ...
```

Requirements:

- Assert that `horizon_steps == attacker_control_sequence.shape[0]`.
- Add batch and candidate dimensions.
- Delegate to `rollout_control_sequences`.
- Do not maintain a second rollout implementation.

## 12. BRAT Reach-Avoid Scoring

The current problem is not a one-set BRT. A valid trajectory score must encode reaching the goal before entering the forbidden capture set.

For trajectory states `states`:

```python
reach = dynamics.reach_fn(states)
failure = -dynamics.avoid_fn(states)
```

Conventions:

- `reach <= 0`: the attacker has reached its success set.
- `failure > 0`: forbidden capture has occurred while the defender remains outside its exclusion radius.

The fixed-trajectory reach-avoid value is:

$$
J_h = \min_{\tau \ge h}\max\left(l_R(x_\tau),\max_{s=h,\ldots,\tau}g(x_s)\right).
$$

Compute all suffix values in linear time using dynamic programming:

```python
def reach_avoid_suffix_values(
    dynamics,
    states: torch.Tensor,              # [...,H+1,S]
    terminal_values: torch.Tensor | None = None,
) -> torch.Tensor:                     # [...,H+1]
    reach = dynamics.reach_fn(states)
    failure = -dynamics.avoid_fn(states)
    result = torch.empty_like(reach)

    if terminal_values is None:
        result[..., -1] = torch.maximum(reach[..., -1], failure[..., -1])
    else:
        result[..., -1] = torch.maximum(
            failure[..., -1],
            torch.minimum(reach[..., -1], terminal_values),
        )

    for k in reversed(range(states.shape[-2] - 1)):
        result[..., k] = torch.maximum(
            failure[..., k],
            torch.minimum(reach[..., k], result[..., k + 1]),
        )

    return result
```

The trajectory score is:

```python
scores = suffix_values[..., 0]
```

This recurrence is equivalent to the reach-avoid min/max temporal objective, but avoids an `O(H^2)` suffix loop.

### 12.1 Event Precedence

`CrazyflieInterception.reach_fn` treats defender entry into the target exclusion region as attacker success. `avoid_fn` suppresses capture failure once the defender has breached that exclusion radius. The scoring function must use these methods unchanged so its event precedence matches the PDE and simulation termination logic.

## 13. Candidate Evaluation and Selection

Implement:

```python
def evaluate_rollouts(
    dynamics,
    rollout: MPCRollout,
    terminal_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scores [B,N] and suffix values [B,N,H+1]."""
```

Implement a generic gather helper:

```python
def gather_candidates(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one candidate per batch from a tensor whose candidate axis is 1."""
```

Do not use Python loops over candidates. Candidate selection must remain GPU-vectorized.

## 14. Iterative MPC Optimization

Implement:

```python
def optimize_control_sequence(
    initial_states: torch.Tensor,       # [B,S]
    initial_times: torch.Tensor,        # [B]
    nominal_controls: torch.Tensor,     # [B,H,U]
    responder: NeuralBangBangController,
    dynamics,
    config: MPCConfig,
    generator: torch.Generator | None = None,
    terminal_value_fn=None,
) -> MPCResult:
    ...
```

### 14.1 Algorithm

For every refinement iteration:

1. Call `sample_control_sequences` around the current nominal sequence.
2. Call `rollout_control_sequences` for all candidates.
3. If using a truncated horizon, evaluate `terminal_value_fn` at final states and remaining times.
4. Call `evaluate_rollouts`.
5. Compute `best_indices = scores.argmin(dim=1)`.
6. Gather the best attacker sequence, states, defender actions, NN values and suffix values.
7. Replace the nominal sequence with the best attacker sequence.
8. Record the best score for diagnostics.
9. Repeat around the updated nominal.

Return the final best sequence and rollout. Since candidate zero is always the current nominal, the best score must be monotonically non-increasing across iterations.

### 14.2 Terminal Values

If a rollout reaches `tau=0`, use:

```python
terminal_values = dynamics.boundary_fn(final_states)
```

If MPC deliberately uses a shorter horizon than the remaining time-to-go, query a frozen network snapshot at the final state and remaining time and use that value as continuation cost.

Do not use a live, changing network inside one dataset-generation pass.

### 14.3 Candidate Chunking

The full `[B,N,H,S]` rollout and neural autograd activations can exceed GPU memory. If `candidate_chunk_size` is configured:

1. Divide candidate axis `N` into chunks.
2. Roll out and score one chunk at a time.
3. Retain only the best candidate and score per batch from each chunk.
4. Compare chunk winners to obtain the global winner.
5. Ensure candidate zero appears in the first chunk.

Time steps remain sequential because the defender policy is closed loop. Initial states and candidates are the dimensions that should be parallelized.

## 15. Receding-Horizon Warm Start

Implement:

```python
def shift_control_sequence(sequence: torch.Tensor) -> torch.Tensor:
    """Drop the first action and repeat the final action."""
    return torch.cat((sequence[..., 1:, :], sequence[..., -1:, :]), dim=-2)
```

Simulation procedure:

1. Initialize nominal controls to zero with shape `[1,H,U]`.
2. Run `optimize_control_sequence`.
3. Apply only `result.controls[0,0]` to the real attacker.
4. Apply `result.defender_controls[0,0]` to the real defender.
5. Advance the real joint state by one simulation step.
6. Shift the winning attacker sequence with `shift_control_sequence`.
7. Use the shifted sequence as the next nominal.
8. Replan from the new state.

This is receding-horizon MPC. Do not execute the complete open-loop winner without replanning.

## 16. `simulate_nn.py` Refactor

### 16.1 Add CLI Arguments

```text
--controller {bang_bang,mpc}
--mpc_horizon_steps
--mpc_num_samples
--mpc_iterations
--mpc_noise_std
--mpc_integrator {euler,rk4}
--mpc_seed
--mpc_candidate_chunk_size
```

Suggested initial defaults for debugging, not final experiments:

```text
mpc_horizon_steps = 25
mpc_num_samples = 128
mpc_iterations = 5
mpc_noise_std = 1.0
mpc_integrator = euler
```

### 16.2 Remove From `simulate_nn.py`

- Local `nn_control` implementation.
- Local `step_dynamics` implementation.
- Embedded bang-bang policy calculations.
- MPC sampling or scoring details.

### 16.3 Keep In `simulate_nn.py`

- Argument parsing.
- Loading `orig_opt.pickle`.
- Dynamics construction.
- Model construction and checkpoint loading.
- Initial conditions.
- Physical problem constants.
- Main simulation loop.
- Termination checks.
- Training-domain warnings.
- Plot generation and saving.

Wrap execution in:

```python
def main():
    ...


if __name__ == "__main__":
    main()
```

This allows tests and training utilities to import simulation helpers without running a simulation.

### 16.4 Bang-Bang Mode

For `--controller bang_bang`:

1. Query `NeuralBangBangController` at the current state and time-to-go.
2. Apply both returned player actions.
3. Integrate one step.

This mode must preserve existing simulation behavior.

### 16.5 MPC Mode

For `--controller mpc`:

1. Optimize the attacker sequence.
2. Apply the first attacker action from the MPC result.
3. Apply the first defender response from the same result.
4. Integrate one real step.
5. Shift the best attacker sequence and repeat.

Update plot titles to report whether the attacker used NN bang-bang or MPC. The defender remains NN bang-bang in both modes.

## 17. Training Data Storage

Create `utils/mpc_data.py`:

```python
class MPCReplayBuffer:
    def __init__(self, state_dim: int, capacity: int): ...

    def __len__(self) -> int: ...

    def add(
        self,
        times: torch.Tensor,       # [M], real time-to-go
        states: torch.Tensor,      # [M,S], real units
        values: torch.Tensor,      # [M]
    ) -> None: ...

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...
```

Requirements:

- Store detached tensors on CPU.
- Enforce a fixed capacity with FIFO replacement.
- Validate finite times, states and values.
- Preserve real units. Normalize only immediately before model evaluation.
- Include replay-buffer state in resumable training checkpoints.

## 18. Bootstrapped MPC Labels

For each winning trajectory:

1. Compute `suffix_values = reach_avoid_suffix_values(...)`.
2. Generate `tau_k = max(tau_initial - k*dt, 0)`.
3. Store every valid triplet `(tau_k, state_k, suffix_value_k)`.
4. Exclude padded/frozen steps after the sample's actual horizon.
5. Optionally downsample adjacent points if highly correlated labels dominate memory.

This implements the paper's trajectory bootstrapping principle while using the correct BRAT objective.

## 19. Training Loss Integration

Do not route the new MPC labels through the existing hard-coded CSL `scenario_optimization` path. Add a clean MPC replay-buffer branch to `experiments.DeepReach.train`.

For an MPC batch:

```python
real_coords = torch.cat((times.unsqueeze(-1), states), dim=-1)
model_coords = dynamics.coord_to_input(real_coords)
results = model({"coords": model_coords})
predicted_values = dynamics.io_to_value(
    results["model_in"],
    results["model_out"].squeeze(-1),
)
mpc_loss = torch.mean((predicted_values - label_values) ** 2)
```

Combine it with the existing BRAT PDE loss:

$$
L_{total}=L_{BRAT-PDE}+\lambda_{MPC}L_{MPC}.
$$

Do not replace the PDE residual. MPC labels are approximate and policy-conditioned; the PDE term provides the global dynamic-programming structure.

Initially implement a fixed `mpc_loss_weight`. Dynamic gradient balancing can be added after the fixed-weight path is tested.

## 20. Training CLI Configuration

Add options to `run_experiment.py` only when MPC supervision is enabled:

```text
--use_mpc_guidance
--mpc_dt
--mpc_horizon_steps
--mpc_num_samples
--mpc_iterations
--mpc_noise_std
--mpc_initial_states
--mpc_refresh_epochs
--mpc_replay_capacity
--mpc_batch_size
--mpc_loss_weight
--mpc_candidate_chunk_size
--mpc_seed
```

Validate that:

- MPC guidance is used only with dynamics implementing `reach_fn`, `avoid_fn`, `optimal_control`, and `optimal_disturbance`.
- `mpc_dt` divides the intended refinement horizon within numerical tolerance.
- Batch size does not exceed the nonempty replay buffer.
- The configured control bounds can be obtained from the dynamics instance.

## 21. Dataset Generation and Refinement Schedule

The two-player problem creates a bootstrapping issue: before a defender network exists, there is no neural defender policy against which attacker MPC can roll out.

For this repository, use the following practical schedule:

1. Complete boundary pretraining and an initial BRAT PDE curriculum segment.
2. Freeze a snapshot of the partially trained network.
3. Sample initial states and time-to-go values in the current curriculum interval.
4. Run batched attacker MPC against the frozen neural defender.
5. Add winning-trajectory suffix labels to the replay buffer.
6. Continue training with combined BRAT PDE and MPC losses.
7. Refresh labels every `mpc_refresh_epochs` using a new frozen snapshot.
8. Keep old and new labels up to replay capacity, or intentionally replace labels from obsolete policies.

When generating labels:

- Set the snapshot to evaluation mode.
- Disable parameter gradients.
- Keep coordinate autograd enabled for policy gradients.
- Detach every rollout step.
- Move completed labels to CPU.
- Restore the training model's prior mode and gradient settings.

## 22. Domain Handling

The neural policy is unreliable outside its sampled training domain.

For every candidate state, compare against:

```python
lower = dynamics.state_mean - dynamics.state_var
upper = dynamics.state_mean + dynamics.state_var
```

Do not clamp the state. Instead support one explicit policy:

- Mark candidate invalid and assign an attacker-worst score, or
- Add a configurable out-of-domain penalty, or
- Permit extrapolation but report the fraction of invalid candidate steps.

Use invalid-candidate rejection for training-label generation. Never train on labels produced after the trajectory leaves the modeled domain unless this is deliberately enabled.

## 23. Numerical and Performance Requirements

1. Vectorize initial states and candidates.
2. Keep only the horizon loop in Python.
3. Avoid per-candidate Python loops.
4. Detach states after every policy query/integration step.
5. Never call `.cpu()` inside the rollout loop.
6. Use candidate chunking when memory is insufficient.
7. Use `torch.Generator` for deterministic sampling.
8. Log the seed and all MPC configuration values.
9. Check every generated value and state for NaN or infinity.
10. Measure generation throughput in trajectories and labels per second.

## 24. Important Limitation

This implementation computes:

> The best sampled attacker response against the current learned defender policy.

It does not compute an exact nested min-max MPC solution. Therefore, its labels are policy-conditioned approximations, not guaranteed game-optimal BRAT values.

A true game-MPC extension would require, for each attacker candidate, maximizing over defender control sequences or running a nested optimizer. That is substantially more expensive. The referenced MPC-guided DeepReach paper handles a single control player and does not solve this two-player extension.

The existing BRAT PDE loss must remain active to preserve the intended differential-game formulation.

## 25. Required Unit Tests

### 25.1 Bang-Bang Tests

1. Batched query returns exact documented shapes.
2. Scalar time broadcasts correctly.
3. Mismatched time batch raises `ValueError`.
4. Controls remain within attacker limits.
5. Disturbances remain within defender limits.
6. Known gradient signs produce the expected bang-bang signs.

### 25.2 Sampling Tests

1. Output shape is `[B,N,H,U]`.
2. Candidate zero equals nominal exactly.
3. Every sampled action respects bounds.
4. A fixed generator seed reproduces identical samples.
5. Zero noise returns copies of nominal.

### 25.3 Integration Tests

1. Euler matches a manually calculated one-step update.
2. RK4 matches analytical constant-acceleration double-integrator motion.
3. Batch dimensions are preserved.
4. Wrapped-state processing is invoked.

### 25.4 Rollout Tests

1. State trajectory shape is `[B,N,H+1,S]`.
2. Defender-control shape is `[B,N,H,D]`.
3. Time-to-go decreases by `dt` and never becomes negative.
4. Expired trajectories freeze correctly.
5. Defender policy is queried at every active timestep.
6. CPU and CUDA results agree within tolerance when CUDA is available.

### 25.5 BRAT Score Tests

Construct deterministic trajectories for:

1. Attacker reaches target before capture.
2. Attacker is captured before reaching target.
3. Defender enters its exclusion region before capture.
4. Neither event occurs before horizon expiry.
5. Capture and defender exclusion occur at the same sampled state.
6. Every bootstrapped suffix value equals direct evaluation of that suffix.

### 25.6 Optimizer Tests

1. MPC selects a known better candidate in a deterministic toy system.
2. Best score never increases between refinement rounds.
3. Returned controls correspond to returned states.
4. Candidate chunking returns the same winner as unchunked evaluation.
5. Receding-horizon shifting preserves shape and repeats the final action.

### 25.7 Training Tests

1. Replay-buffer capacity and FIFO behavior are correct.
2. Replay-buffer checkpoint round-trip is exact.
3. MPC labels remain in real units.
4. Model coordinates are normalized exactly once.
5. MPC loss backpropagates into model parameters.
6. BRAT PDE and MPC losses can be optimized in one step.
7. Label generation leaves no model parameter gradients allocated.
8. Resume restores replay data and generation counters.

## 26. End-to-End Acceptance Checks

The implementation is complete only when all of the following pass:

1. `simulate_nn.py --controller bang_bang` reproduces the existing controller behavior.
2. A small CPU MPC run completes without shape, device, or autograd errors.
3. A CUDA MPC run processes candidates in parallel when CUDA is available.
4. The MPC winner's score is no worse than the nominal candidate's score.
5. Receding-horizon simulation applies the first optimized action and replans.
6. MPC labels are generated for every valid state on winning trajectories.
7. Combined BRAT PDE/MPC training completes at least one optimizer step.
8. Training can checkpoint and resume with the MPC replay buffer intact.
9. Out-of-domain candidate trajectories are detected and excluded from labels.
10. Logs report MPC configuration, generation throughput, score statistics, and replay-buffer size.

## 27. Recommended Implementation Order

1. Complete and test `controllers/bang_bang.py`.
2. Implement and test `sample_control_sequences`.
3. Implement and test `integrate_step`.
4. Implement and test batched closed-loop rollout.
5. Implement and test BRAT suffix scoring.
6. Implement iterative MPC selection and candidate chunking.
7. Refactor `simulate_nn.py` and validate both controller modes.
8. Add `MPCReplayBuffer` and trajectory bootstrapping.
9. Add combined MPC/BRAT training loss.
10. Add periodic dataset refinement and checkpoint persistence.
11. Run end-to-end simulation and training smoke tests.
