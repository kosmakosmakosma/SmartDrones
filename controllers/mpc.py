from dataclasses import dataclass
from typing import Callable, Optional

import torch

from controllers.bang_bang import NeuralBangBangController


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
    candidate_chunk_size: Optional[int] = None
    control_hold_steps: int = 1
    include_axis_candidates: bool = True

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if self.integration_method not in ("euler", "rk4"):
            raise ValueError("integration_method must be 'euler' or 'rk4'")
        lower = torch.as_tensor(self.control_lower, dtype=torch.float32)
        upper = torch.as_tensor(self.control_upper, dtype=torch.float32)
        if torch.any(lower >= upper):
            raise ValueError("control_lower must be strictly less than control_upper")
        if self.candidate_chunk_size is not None and self.candidate_chunk_size < 1:
            raise ValueError("candidate_chunk_size must be >= 1 if provided")
        if self.control_hold_steps < 1:
            raise ValueError("control_hold_steps must be >= 1")


@dataclass
class MPCRollout:
    states: torch.Tensor            # [B,N,H+1,S]
    attacker_controls: torch.Tensor  # [B,N,H,U]
    defender_controls: torch.Tensor  # [B,N,H,D]
    network_values: torch.Tensor    # [B,N,H+1]
    times: torch.Tensor             # [B,N,H+1]
    valid_steps: Optional[torch.Tensor] = None  # [B,N,H]


@dataclass
class MPCResult:
    controls: torch.Tensor          # [B,H,U]
    states: torch.Tensor            # [B,H+1,S]
    defender_controls: torch.Tensor  # [B,H,D]
    network_values: torch.Tensor    # [B,H+1]
    suffix_values: torch.Tensor     # [B,H+1]
    score: torch.Tensor             # [B]
    all_scores: torch.Tensor        # [B,N] (final iteration, all candidates)


def sample_control_sequences(
    nominal, num_samples, noise_std, lower, upper, generator=None, control_hold_steps=1,
    include_axis_candidates=False,
):
    """Gaussian perturbations of `nominal` [B,H,U] -> [B,N,H,U]; candidate 0 == nominal."""
    if nominal.ndim != 3:
        raise ValueError("nominal must have shape [B,H,U]")
    batch_size, horizon_steps, control_dim = nominal.shape
    device, dtype = nominal.device, nominal.dtype

    lower = torch.as_tensor(lower, dtype=dtype, device=device).reshape(control_dim)
    upper = torch.as_tensor(upper, dtype=dtype, device=device).reshape(control_dim)

    expanded = nominal.unsqueeze(1).expand(batch_size, num_samples, horizon_steps, control_dim).clone()
    num_control_blocks = (horizon_steps + control_hold_steps - 1) // control_hold_steps
    block_noise = torch.randn(
        batch_size, num_samples, num_control_blocks, control_dim,
        generator=generator, device=device, dtype=dtype,
    ) * noise_std
    noise = block_noise.repeat_interleave(control_hold_steps, dim=2)[:, :, :horizon_steps]
    samples = expanded + noise

    lower_b = lower.view(1, 1, 1, control_dim)
    upper_b = upper.view(1, 1, 1, control_dim)
    samples = torch.maximum(torch.minimum(samples, upper_b), lower_b)

    samples[:, 0] = nominal
    if include_axis_candidates:
        baseline = torch.maximum(torch.minimum(torch.zeros_like(nominal), upper), lower)
        candidate_index = 1
        for control_index in range(control_dim):
            for bound in (lower[control_index], upper[control_index]):
                if candidate_index >= num_samples:
                    break
                samples[:, candidate_index] = baseline
                samples[:, candidate_index, :, control_index] = bound
                candidate_index += 1
    return samples


def integrate_step(dynamics, states, controls, disturbances, dt, method):
    """One dynamics step; supports 'euler' and 'rk4', with fixed controls/disturbances per step."""
    if method == "euler":
        next_states = states + dt * dynamics.dsdt(states, controls, disturbances)
    elif method == "rk4":
        k1 = dynamics.dsdt(states, controls, disturbances)
        k2 = dynamics.dsdt(states + 0.5 * dt * k1, controls, disturbances)
        k3 = dynamics.dsdt(states + 0.5 * dt * k2, controls, disturbances)
        k4 = dynamics.dsdt(states + dt * k3, controls, disturbances)
        next_states = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        raise ValueError(f"unknown integration method '{method}'")
    return dynamics.equivalent_wrapped_state(next_states)


def rollout_control_sequences(
    initial_states, initial_times, attacker_controls,
    responder: NeuralBangBangController, dynamics, dt, integration_method="euler",
):
    """Closed-loop rollout: attacker follows `attacker_controls`, defender follows `responder`."""
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [B,S]")
    if attacker_controls.ndim != 4:
        raise ValueError("attacker_controls must have shape [B,N,H,U]")

    batch_size, state_dim = initial_states.shape
    controls_batch_size, num_candidates, horizon_steps, control_dim = attacker_controls.shape
    if controls_batch_size != batch_size:
        raise ValueError("batch size mismatch between initial_states and attacker_controls")

    device, dtype = initial_states.device, initial_states.dtype

    initial_times = torch.as_tensor(initial_times, dtype=dtype, device=device).reshape(-1)
    if initial_times.numel() == 1:
        initial_times = initial_times.expand(batch_size)
    if initial_times.shape[0] != batch_size:
        raise ValueError("initial_times must be scalar or one value per batch element")

    disturbance_dim = dynamics.disturbance_dim
    states = torch.zeros(batch_size, num_candidates, horizon_steps + 1, state_dim, device=device, dtype=dtype)
    defender_controls = torch.zeros(batch_size, num_candidates, horizon_steps, disturbance_dim, device=device, dtype=dtype)
    network_values = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    times = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    valid_steps = torch.zeros(batch_size, num_candidates, horizon_steps, dtype=torch.bool, device=device)

    current_states = initial_states.unsqueeze(1).expand(batch_size, num_candidates, state_dim).clone()
    current_times = initial_times.unsqueeze(1).expand(batch_size, num_candidates).clone()
    states[:, :, 0] = current_states
    times[:, :, 0] = current_times

    for step in range(horizon_steps):
        flat_states = current_states.reshape(batch_size * num_candidates, state_dim)
        flat_times = current_times.reshape(batch_size * num_candidates)

        query = responder.query(flat_states, flat_times)
        network_values[:, :, step] = query.values.reshape(batch_size, num_candidates)
        disturbances = query.disturbances.reshape(batch_size, num_candidates, disturbance_dim)

        controls_step = attacker_controls[:, :, step, :]

        flat_controls = controls_step.reshape(batch_size * num_candidates, control_dim)
        flat_disturbances = disturbances.reshape(batch_size * num_candidates, disturbance_dim)
        next_flat_states = integrate_step(dynamics, flat_states, flat_controls, flat_disturbances, dt, integration_method)
        next_states = next_flat_states.reshape(batch_size, num_candidates, state_dim).detach()

        active = current_times > 0
        next_states = torch.where(active.unsqueeze(-1), next_states, current_states)
        next_times = torch.clamp(current_times - dt, min=0.0)

        current_states, current_times = next_states, next_times

        states[:, :, step + 1] = current_states
        defender_controls[:, :, step] = disturbances
        times[:, :, step + 1] = current_times
        valid_steps[:, :, step] = active

    flat_states = current_states.reshape(batch_size * num_candidates, state_dim)
    flat_times = current_times.reshape(batch_size * num_candidates)
    final_query = responder.query(flat_states, flat_times)
    network_values[:, :, horizon_steps] = final_query.values.reshape(batch_size, num_candidates)

    return MPCRollout(
        states=states,
        attacker_controls=attacker_controls,
        defender_controls=defender_controls,
        network_values=network_values,
        times=times,
        valid_steps=valid_steps,
    )


def rollout_disturbance_sequences(
    initial_states, initial_times, defender_controls,
    responder: NeuralBangBangController, dynamics, dt, integration_method="euler",
):
    """Closed-loop rollout: defender follows sampled sequences, attacker follows `responder`."""
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [B,S]")
    if defender_controls.ndim != 4:
        raise ValueError("defender_controls must have shape [B,N,H,D]")

    batch_size, state_dim = initial_states.shape
    controls_batch_size, num_candidates, horizon_steps, disturbance_dim = defender_controls.shape
    if controls_batch_size != batch_size:
        raise ValueError("batch size mismatch between initial_states and defender_controls")

    device, dtype = initial_states.device, initial_states.dtype
    initial_times = torch.as_tensor(initial_times, dtype=dtype, device=device).reshape(-1)
    if initial_times.numel() == 1:
        initial_times = initial_times.expand(batch_size)
    if initial_times.shape[0] != batch_size:
        raise ValueError("initial_times must be scalar or one value per batch element")

    control_dim = dynamics.control_dim
    states = torch.zeros(batch_size, num_candidates, horizon_steps + 1, state_dim, device=device, dtype=dtype)
    attacker_controls = torch.zeros(batch_size, num_candidates, horizon_steps, control_dim, device=device, dtype=dtype)
    network_values = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    times = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    valid_steps = torch.zeros(batch_size, num_candidates, horizon_steps, dtype=torch.bool, device=device)

    current_states = initial_states.unsqueeze(1).expand(batch_size, num_candidates, state_dim).clone()
    current_times = initial_times.unsqueeze(1).expand(batch_size, num_candidates).clone()
    states[:, :, 0] = current_states
    times[:, :, 0] = current_times

    for step in range(horizon_steps):
        flat_states = current_states.reshape(batch_size * num_candidates, state_dim)
        flat_times = current_times.reshape(batch_size * num_candidates)
        query = responder.query(flat_states, flat_times)
        network_values[:, :, step] = query.values.reshape(batch_size, num_candidates)

        controls = query.controls.reshape(batch_size, num_candidates, control_dim)
        disturbances = defender_controls[:, :, step, :]
        next_flat_states = integrate_step(
            dynamics,
            flat_states,
            controls.reshape(batch_size * num_candidates, control_dim),
            disturbances.reshape(batch_size * num_candidates, disturbance_dim),
            dt,
            integration_method,
        )
        next_states = next_flat_states.reshape(batch_size, num_candidates, state_dim).detach()

        active = current_times > 0
        current_states = torch.where(active.unsqueeze(-1), next_states, current_states)
        current_times = torch.clamp(current_times - dt, min=0.0)

        states[:, :, step + 1] = current_states
        attacker_controls[:, :, step] = controls
        times[:, :, step + 1] = current_times
        valid_steps[:, :, step] = active

    final_query = responder.query(
        current_states.reshape(batch_size * num_candidates, state_dim),
        current_times.reshape(batch_size * num_candidates),
    )
    network_values[:, :, horizon_steps] = final_query.values.reshape(batch_size, num_candidates)

    return MPCRollout(
        states=states,
        attacker_controls=attacker_controls,
        defender_controls=defender_controls,
        network_values=network_values,
        times=times,
        valid_steps=valid_steps,
    )


def rollout_joint_sequences(
    initial_states, initial_times, attacker_controls, defender_controls,
    responder: NeuralBangBangController, dynamics, dt, integration_method="euler",
):
    """Roll out paired attacker and defender sequences with shape [B,N,H,*]."""
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [B,S]")
    if attacker_controls.ndim != 4 or defender_controls.ndim != 4:
        raise ValueError("control sequences must have shape [B,N,H,*]")
    if attacker_controls.shape[:3] != defender_controls.shape[:3]:
        raise ValueError("attacker and defender sequence dimensions must match")

    batch_size, state_dim = initial_states.shape
    controls_batch_size, num_candidates, horizon_steps, control_dim = attacker_controls.shape
    disturbance_dim = defender_controls.shape[-1]
    if controls_batch_size != batch_size:
        raise ValueError("batch size mismatch between initial states and controls")
    if control_dim != dynamics.control_dim or disturbance_dim != dynamics.disturbance_dim:
        raise ValueError("control dimensions do not match dynamics")

    device, dtype = initial_states.device, initial_states.dtype
    initial_times = torch.as_tensor(initial_times, dtype=dtype, device=device).reshape(-1)
    if initial_times.numel() == 1:
        initial_times = initial_times.expand(batch_size)
    if initial_times.shape[0] != batch_size:
        raise ValueError("initial_times must be scalar or one value per batch element")

    states = torch.zeros(batch_size, num_candidates, horizon_steps + 1, state_dim, device=device, dtype=dtype)
    network_values = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    times = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    valid_steps = torch.zeros(batch_size, num_candidates, horizon_steps, dtype=torch.bool, device=device)

    current_states = initial_states.unsqueeze(1).expand(batch_size, num_candidates, state_dim).clone()
    current_times = initial_times.unsqueeze(1).expand(batch_size, num_candidates).clone()
    states[:, :, 0] = current_states
    times[:, :, 0] = current_times

    for step in range(horizon_steps):
        flat_states = current_states.reshape(batch_size * num_candidates, state_dim)
        flat_times = current_times.reshape(batch_size * num_candidates)
        query = responder.query(flat_states, flat_times)
        network_values[:, :, step] = query.values.reshape(batch_size, num_candidates)

        next_flat_states = integrate_step(
            dynamics,
            flat_states,
            attacker_controls[:, :, step].reshape(batch_size * num_candidates, control_dim),
            defender_controls[:, :, step].reshape(batch_size * num_candidates, disturbance_dim),
            dt,
            integration_method,
        )
        next_states = next_flat_states.reshape(batch_size, num_candidates, state_dim).detach()
        active = current_times > 0
        current_states = torch.where(active.unsqueeze(-1), next_states, current_states)
        current_times = torch.clamp(current_times - dt, min=0.0)

        states[:, :, step + 1] = current_states
        times[:, :, step + 1] = current_times
        valid_steps[:, :, step] = active

    final_query = responder.query(
        current_states.reshape(batch_size * num_candidates, state_dim),
        current_times.reshape(batch_size * num_candidates),
    )
    network_values[:, :, horizon_steps] = final_query.values.reshape(batch_size, num_candidates)

    return MPCRollout(
        states=states,
        attacker_controls=attacker_controls,
        defender_controls=defender_controls,
        network_values=network_values,
        times=times,
        valid_steps=valid_steps,
    )


def rollout_trajectory(
    initial_state, horizon_steps, attacker_control_sequence, initial_time,
    responder, dynamics, dt, integration_method="euler",
):
    """Single-trajectory convenience wrapper: initial state + control sequence -> state sequence."""
    if attacker_control_sequence.shape[0] != horizon_steps:
        raise ValueError("attacker_control_sequence length must equal horizon_steps")

    state = torch.as_tensor(initial_state, dtype=torch.float32)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[0] != 1:
        raise ValueError("rollout_trajectory expects a single initial state")

    controls = attacker_control_sequence.unsqueeze(0).unsqueeze(0).to(dtype=state.dtype, device=state.device)
    times = torch.as_tensor(initial_time, dtype=state.dtype, device=state.device).reshape(1)

    return rollout_control_sequences(state, times, controls, responder, dynamics, dt, integration_method)


def reach_avoid_suffix_values(dynamics, states, terminal_values=None):
    """Backward recursion of the BRAT reach-avoid objective over trajectory suffixes.

    states: [...,H+1,S] -> returns [...,H+1], where index k is
    min_{tau>=k} max(reach(x_tau), max_{s=k..tau} -avoid(x_s)).
    """
    reach = dynamics.reach_fn(states)
    failure = -dynamics.avoid_fn(states)
    result = torch.empty_like(reach)

    if terminal_values is None:
        result[..., -1] = torch.maximum(reach[..., -1], failure[..., -1])
    else:
        result[..., -1] = torch.maximum(failure[..., -1], torch.minimum(reach[..., -1], terminal_values))

    for step in reversed(range(states.shape[-2] - 1)):
        result[..., step] = torch.maximum(failure[..., step], torch.minimum(reach[..., step], result[..., step + 1]))

    return result


def evaluate_rollouts(dynamics, rollout: MPCRollout, terminal_values=None):
    """Returns (scores [B,N], suffix_values [B,N,H+1]); lower score is better for the attacker."""
    suffix_values = reach_avoid_suffix_values(dynamics, rollout.states, terminal_values)
    scores = suffix_values[..., 0]
    return scores, suffix_values


def gather_candidates(tensor, indices):
    """Gather one candidate per batch element from `tensor` whose dim=1 is the candidate axis."""
    batch_size = tensor.shape[0]
    view_shape = (batch_size,) + (1,) * (tensor.ndim - 1)
    expand_shape = (batch_size, 1) + tensor.shape[2:]
    index = indices.view(*view_shape).expand(*expand_shape)
    return torch.gather(tensor, dim=1, index=index).squeeze(1)


def _select_best_candidate(scores, candidate_tensors, maximize=False):
    best_indices = scores.argmax(dim=1) if maximize else scores.argmin(dim=1)
    best_scores = gather_candidates(scores.unsqueeze(-1), best_indices).squeeze(-1)
    gathered = {name: gather_candidates(tensor, best_indices) for name, tensor in candidate_tensors.items()}
    return best_scores, gathered


def _masked_update(old, new, mask):
    view_shape = (mask.shape[0],) + (1,) * (old.ndim - 1)
    return torch.where(mask.view(*view_shape), new, old)


def optimize_control_sequence(
    initial_states, initial_times, nominal_controls,
    responder: NeuralBangBangController, dynamics, config: MPCConfig,
    generator: Optional[torch.Generator] = None,
    terminal_value_fn: Optional[Callable] = None,
    use_network_terminal_value: bool = False,
) -> MPCResult:
    """Iterative sampling-based MPC: perturb -> rollout -> score -> keep best -> repeat."""
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [B,S]")

    nominal = nominal_controls
    all_scores = None
    best = None
    best_score = None

    for _ in range(config.num_iterations):
        candidates = sample_control_sequences(
            nominal, config.num_samples, config.noise_std,
            config.control_lower, config.control_upper, generator=generator,
            control_hold_steps=config.control_hold_steps,
            include_axis_candidates=config.include_axis_candidates,
        )

        chunk_size = config.candidate_chunk_size or config.num_samples
        chunk_scores = []
        iter_best_score = None
        iter_best = None

        for start in range(0, config.num_samples, chunk_size):
            end = min(start + chunk_size, config.num_samples)
            chunk_controls = candidates[:, start:end]

            rollout = rollout_control_sequences(
                initial_states, initial_times, chunk_controls,
                responder, dynamics, config.dt, config.integration_method,
            )

            terminal_values = None
            if terminal_value_fn is not None:
                terminal_values = terminal_value_fn(rollout.states[:, :, -1], rollout.times[:, :, -1])
            elif use_network_terminal_value:
                terminal_values = rollout.network_values[:, :, -1]

            scores, suffix_values = evaluate_rollouts(dynamics, rollout, terminal_values)
            chunk_scores.append(scores)

            chunk_best_score, chunk_best = _select_best_candidate(scores, {
                "controls": chunk_controls,
                "states": rollout.states,
                "defender_controls": rollout.defender_controls,
                "network_values": rollout.network_values,
                "suffix_values": suffix_values,
            })

            if iter_best_score is None:
                iter_best_score, iter_best = chunk_best_score, chunk_best
            else:
                improved = chunk_best_score < iter_best_score
                iter_best_score = torch.where(improved, chunk_best_score, iter_best_score)
                iter_best = {key: _masked_update(iter_best[key], chunk_best[key], improved) for key in iter_best}

        all_scores = torch.cat(chunk_scores, dim=1)
        nominal = iter_best["controls"]
        best_score, best = iter_best_score, iter_best

    return MPCResult(
        controls=best["controls"],
        states=best["states"],
        defender_controls=best["defender_controls"],
        network_values=best["network_values"],
        suffix_values=best["suffix_values"],
        score=best_score,
        all_scores=all_scores,
    )


def optimize_disturbance_sequence(
    initial_states, initial_times, nominal_disturbances,
    responder: NeuralBangBangController, dynamics, config: MPCConfig,
    generator: Optional[torch.Generator] = None,
    terminal_value_fn: Optional[Callable] = None,
    use_network_terminal_value: bool = False,
) -> MPCResult:
    """Sampling-based defender MPC; the defender maximizes the BRAT value."""
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [B,S]")

    nominal = nominal_disturbances
    all_scores = None
    best = None
    best_score = None

    for _ in range(config.num_iterations):
        candidates = sample_control_sequences(
            nominal, config.num_samples, config.noise_std,
            config.control_lower, config.control_upper, generator=generator,
            control_hold_steps=config.control_hold_steps,
            include_axis_candidates=config.include_axis_candidates,
        )
        chunk_size = config.candidate_chunk_size or config.num_samples
        chunk_scores = []
        iter_best_score = None
        iter_best = None

        for start in range(0, config.num_samples, chunk_size):
            end = min(start + chunk_size, config.num_samples)
            chunk_disturbances = candidates[:, start:end]
            rollout = rollout_disturbance_sequences(
                initial_states, initial_times, chunk_disturbances,
                responder, dynamics, config.dt, config.integration_method,
            )

            terminal_values = None
            if terminal_value_fn is not None:
                terminal_values = terminal_value_fn(rollout.states[:, :, -1], rollout.times[:, :, -1])
            elif use_network_terminal_value:
                terminal_values = rollout.network_values[:, :, -1]

            scores, suffix_values = evaluate_rollouts(dynamics, rollout, terminal_values)
            chunk_scores.append(scores)
            chunk_best_score, chunk_best = _select_best_candidate(scores, {
                "controls": rollout.attacker_controls,
                "states": rollout.states,
                "defender_controls": chunk_disturbances,
                "network_values": rollout.network_values,
                "suffix_values": suffix_values,
            }, maximize=True)

            if iter_best_score is None:
                iter_best_score, iter_best = chunk_best_score, chunk_best
            else:
                improved = chunk_best_score > iter_best_score
                iter_best_score = torch.where(improved, chunk_best_score, iter_best_score)
                iter_best = {key: _masked_update(iter_best[key], chunk_best[key], improved) for key in iter_best}

        all_scores = torch.cat(chunk_scores, dim=1)
        nominal = iter_best["defender_controls"]
        best_score, best = iter_best_score, iter_best

    return MPCResult(
        controls=best["controls"],
        states=best["states"],
        defender_controls=best["defender_controls"],
        network_values=best["network_values"],
        suffix_values=best["suffix_values"],
        score=best_score,
        all_scores=all_scores,
    )


def optimize_joint_sequences(
    initial_states, initial_times, nominal_controls, nominal_disturbances,
    responder: NeuralBangBangController, dynamics,
    attacker_config: MPCConfig, defender_config: MPCConfig,
    generator: Optional[torch.Generator] = None,
    terminal_value_fn: Optional[Callable] = None,
    use_network_terminal_value: bool = False,
) -> MPCResult:
    """Approximate game MPC by alternating sampled attacker and defender best responses."""
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [B,S]")
    compatible_fields = ("dt", "horizon_steps", "num_iterations", "integration_method")
    if any(getattr(attacker_config, field) != getattr(defender_config, field)
           for field in compatible_fields):
        raise ValueError("attacker and defender configs must share dt, horizon, iterations, and integrator")

    attacker_nominal = nominal_controls
    defender_nominal = nominal_disturbances
    all_scores = None
    best = None
    best_score = None
    batch_size = initial_states.shape[0]

    for _ in range(attacker_config.num_iterations):
        attacker_candidates = sample_control_sequences(
            attacker_nominal, attacker_config.num_samples, attacker_config.noise_std,
            attacker_config.control_lower, attacker_config.control_upper, generator=generator,
            control_hold_steps=attacker_config.control_hold_steps,
            include_axis_candidates=attacker_config.include_axis_candidates,
        )
        chunk_size = attacker_config.candidate_chunk_size or attacker_config.num_samples
        iter_best_score = None
        iter_best = None

        for start in range(0, attacker_config.num_samples, chunk_size):
            end = min(start + chunk_size, attacker_config.num_samples)
            chunk_controls = attacker_candidates[:, start:end]
            chunk_disturbances = defender_nominal.unsqueeze(1).expand(
                batch_size, end - start, defender_config.horizon_steps, dynamics.disturbance_dim)
            rollout = rollout_joint_sequences(
                initial_states, initial_times, chunk_controls, chunk_disturbances,
                responder, dynamics, attacker_config.dt, attacker_config.integration_method,
            )
            terminal_values = None
            if terminal_value_fn is not None:
                terminal_values = terminal_value_fn(rollout.states[:, :, -1], rollout.times[:, :, -1])
            elif use_network_terminal_value:
                terminal_values = rollout.network_values[:, :, -1]
            scores, suffix_values = evaluate_rollouts(dynamics, rollout, terminal_values)
            chunk_best_score, chunk_best = _select_best_candidate(scores, {
                "controls": chunk_controls,
                "states": rollout.states,
                "defender_controls": chunk_disturbances,
                "network_values": rollout.network_values,
                "suffix_values": suffix_values,
            })
            if iter_best_score is None:
                iter_best_score, iter_best = chunk_best_score, chunk_best
            else:
                improved = chunk_best_score < iter_best_score
                iter_best_score = torch.where(improved, chunk_best_score, iter_best_score)
                iter_best = {key: _masked_update(iter_best[key], chunk_best[key], improved) for key in iter_best}

        attacker_nominal = iter_best["controls"]
        defender_candidates = sample_control_sequences(
            defender_nominal, defender_config.num_samples, defender_config.noise_std,
            defender_config.control_lower, defender_config.control_upper, generator=generator,
            control_hold_steps=defender_config.control_hold_steps,
            include_axis_candidates=defender_config.include_axis_candidates,
        )
        chunk_size = defender_config.candidate_chunk_size or defender_config.num_samples
        chunk_scores = []
        iter_best_score = None
        iter_best = None

        for start in range(0, defender_config.num_samples, chunk_size):
            end = min(start + chunk_size, defender_config.num_samples)
            chunk_disturbances = defender_candidates[:, start:end]
            chunk_controls = attacker_nominal.unsqueeze(1).expand(
                batch_size, end - start, attacker_config.horizon_steps, dynamics.control_dim)
            rollout = rollout_joint_sequences(
                initial_states, initial_times, chunk_controls, chunk_disturbances,
                responder, dynamics, defender_config.dt, defender_config.integration_method,
            )
            terminal_values = None
            if terminal_value_fn is not None:
                terminal_values = terminal_value_fn(rollout.states[:, :, -1], rollout.times[:, :, -1])
            elif use_network_terminal_value:
                terminal_values = rollout.network_values[:, :, -1]
            scores, suffix_values = evaluate_rollouts(dynamics, rollout, terminal_values)
            chunk_scores.append(scores)
            chunk_best_score, chunk_best = _select_best_candidate(scores, {
                "controls": chunk_controls,
                "states": rollout.states,
                "defender_controls": chunk_disturbances,
                "network_values": rollout.network_values,
                "suffix_values": suffix_values,
            }, maximize=True)
            if iter_best_score is None:
                iter_best_score, iter_best = chunk_best_score, chunk_best
            else:
                improved = chunk_best_score > iter_best_score
                iter_best_score = torch.where(improved, chunk_best_score, iter_best_score)
                iter_best = {key: _masked_update(iter_best[key], chunk_best[key], improved) for key in iter_best}

        defender_nominal = iter_best["defender_controls"]
        all_scores = torch.cat(chunk_scores, dim=1)
        best_score, best = iter_best_score, iter_best

    return MPCResult(
        controls=best["controls"],
        states=best["states"],
        defender_controls=best["defender_controls"],
        network_values=best["network_values"],
        suffix_values=best["suffix_values"],
        score=best_score,
        all_scores=all_scores,
    )


def shift_control_sequence(sequence):
    """Receding-horizon warm start: drop the first action, repeat the final action."""
    return torch.cat((sequence[..., 1:, :], sequence[..., -1:, :]), dim=-2)
