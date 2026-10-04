from dataclasses import dataclass, replace
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
    # optional box constraint on this player's own state dims: candidates leaving it before the game ends are rejected
    domain_dims: Optional[tuple] = None
    domain_lower: Optional[torch.Tensor] = None
    domain_upper: Optional[torch.Tensor] = None

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
    event_steps: Optional[torch.Tensor] = None  # [B] first step at which the game ended (H if it never did)
    upper_value: Optional[torch.Tensor] = None  # [B] max-min solver: attacker's guaranteed score min_i max_j
    lower_value: Optional[torch.Tensor] = None  # [B] max-min solver: defender's guaranteed score max_j min_i


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
    responder: Optional[NeuralBangBangController], dynamics, dt, integration_method="euler",
    network_values: str = "all",
):
    """Roll out paired attacker and defender sequences with shape [B,N,H,*].

    Both players follow their sequences, so the network is only needed for recorded values:
    network_values='all' queries it at every step, 'final' only at the last state (the terminal
    value), 'none' never (values stay 0; also used when responder is None).
    """
    if network_values not in ("all", "final", "none"):
        raise ValueError("network_values must be 'all', 'final' or 'none'")
    if responder is None:
        network_values = "none"
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
    values = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    times = torch.zeros(batch_size, num_candidates, horizon_steps + 1, device=device, dtype=dtype)
    valid_steps = torch.zeros(batch_size, num_candidates, horizon_steps, dtype=torch.bool, device=device)

    current_states = initial_states.unsqueeze(1).expand(batch_size, num_candidates, state_dim).clone()
    current_times = initial_times.unsqueeze(1).expand(batch_size, num_candidates).clone()
    states[:, :, 0] = current_states
    times[:, :, 0] = current_times

    for step in range(horizon_steps):
        flat_states = current_states.reshape(batch_size * num_candidates, state_dim)
        flat_times = current_times.reshape(batch_size * num_candidates)
        if network_values == "all":
            query = responder.query(flat_states, flat_times)
            values[:, :, step] = query.values.reshape(batch_size, num_candidates)

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

    if network_values != "none":
        final_query = responder.query(
            current_states.reshape(batch_size * num_candidates, state_dim),
            current_times.reshape(batch_size * num_candidates),
        )
        values[:, :, horizon_steps] = final_query.values.reshape(batch_size, num_candidates)

    return MPCRollout(
        states=states,
        attacker_controls=attacker_controls,
        defender_controls=defender_controls,
        network_values=values,
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


DOMAIN_PENALTY = 1e3


def domain_violation(dynamics, config: MPCConfig, states):
    """[...] largest distance by which config.domain_dims leave [domain_lower, domain_upper] before the game ends.

    states: [...,H+1,S]. Steps after the first state inside the reach or avoid set are ignored, since
    the game is over there. Returns None when the config has no domain constraint.
    """
    if config.domain_dims is None:
        return None
    dims = list(config.domain_dims)
    lower = torch.as_tensor(config.domain_lower, dtype=states.dtype, device=states.device)
    upper = torch.as_tensor(config.domain_upper, dtype=states.dtype, device=states.device)
    own = states[..., dims]
    excess = (torch.clamp(own - upper, min=0) + torch.clamp(lower - own, min=0)).amax(dim=-1)
    ended = ((dynamics.reach_fn(states) <= 0) | (dynamics.avoid_fn(states) <= 0)).int()
    after_end = (torch.cumsum(ended, dim=-1) - ended) > 0
    return excess.masked_fill(after_end, 0.0).amax(dim=-1)


def _selection_scores(dynamics, config, states, scores, maximize):
    """Scores used to pick candidates: domain-violating candidates rank below every valid one
    (least violation first if none is valid). The reported score and labels stay unpenalised."""
    violation = domain_violation(dynamics, config, states)
    if violation is None:
        return scores
    penalty = torch.where(violation > 0, DOMAIN_PENALTY + violation, torch.zeros_like(violation))
    return scores - penalty if maximize else scores + penalty


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

            chunk_best_score, chunk_best = _select_best_candidate(_selection_scores(dynamics, config, rollout.states, scores, False), {
                "score": scores,
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
        score=best["score"],
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
            chunk_best_score, chunk_best = _select_best_candidate(_selection_scores(dynamics, config, rollout.states, scores, True), {
                "score": scores,
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
        score=best["score"],
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
    # Both players follow sequences here, so the network only matters for the terminal value.
    query_mode = "final" if (use_network_terminal_value and terminal_value_fn is None) else "none"
    initial_values = None if responder is None else responder.query(initial_states, initial_times).values

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
                responder, dynamics, attacker_config.dt, attacker_config.integration_method, query_mode,
            )
            if initial_values is not None:
                rollout.network_values[:, :, 0] = initial_values[:, None]
            terminal_values = None
            if terminal_value_fn is not None:
                terminal_values = terminal_value_fn(rollout.states[:, :, -1], rollout.times[:, :, -1])
            elif use_network_terminal_value:
                terminal_values = rollout.network_values[:, :, -1] if responder is not None else None
            scores, suffix_values = evaluate_rollouts(dynamics, rollout, terminal_values)
            chunk_best_score, chunk_best = _select_best_candidate(_selection_scores(dynamics, attacker_config, rollout.states, scores, False), {
                "score": scores,
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
                responder, dynamics, defender_config.dt, defender_config.integration_method, query_mode,
            )
            if initial_values is not None:
                rollout.network_values[:, :, 0] = initial_values[:, None]
            terminal_values = None
            if terminal_value_fn is not None:
                terminal_values = terminal_value_fn(rollout.states[:, :, -1], rollout.times[:, :, -1])
            elif use_network_terminal_value:
                terminal_values = rollout.network_values[:, :, -1] if responder is not None else None
            scores, suffix_values = evaluate_rollouts(dynamics, rollout, terminal_values)
            chunk_scores.append(scores)
            chunk_best_score, chunk_best = _select_best_candidate(_selection_scores(dynamics, defender_config, rollout.states, scores, True), {
                "score": scores,
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
        score=best["score"],
        all_scores=all_scores,
    )


def _violation_penalty(violation):
    return torch.where(violation > 0, DOMAIN_PENALTY + violation, torch.zeros_like(violation))


def optimize_maxmin_sequences(
    initial_states, initial_times, nominal_controls, nominal_disturbances,
    responder: Optional[NeuralBangBangController], dynamics,
    attacker_config: MPCConfig, defender_config: MPCConfig,
    generator: Optional[torch.Generator] = None,
    terminal_value_fn: Optional[Callable] = None,
    use_network_terminal_value: bool = False,
) -> MPCResult:
    """Robust game MPC over the full table of attacker x defender candidate plans.

    Each iteration samples attacker and defender plans around the current ones, rolls out every
    pair and builds the score table M[b, i, j]. The attacker keeps the plan with the best worst
    case, argmin_i max_j M[i, j]; the defender keeps argmax_j min_i M[i, j]. A candidate whose own
    drone leaves its domain ranks last for its owner and is not used as a counterplay by the other
    player (unless no candidate of that player is valid).
    attacker_config.candidate_chunk_size sets how many attacker plans are rolled out against all
    defender plans at once (default 8).
    Returns the rollout of the two chosen plans, with upper_value = min_i max_j M (what the attacker
    guarantees) and lower_value = max_j min_i M (what the defender guarantees) from the last table.
    """
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [B,S]")
    compatible_fields = ("dt", "horizon_steps", "num_iterations", "integration_method")
    if any(getattr(attacker_config, field) != getattr(defender_config, field) for field in compatible_fields):
        raise ValueError("attacker and defender configs must share dt, horizon, iterations, and integrator")

    batch_size = initial_states.shape[0]
    horizon_steps = attacker_config.horizon_steps
    num_attacker, num_defender = attacker_config.num_samples, defender_config.num_samples
    rows_per_chunk = attacker_config.candidate_chunk_size or min(num_attacker, 8)
    use_network = use_network_terminal_value and terminal_value_fn is None and responder is not None
    query_mode = "final" if use_network else "none"
    batch_index = torch.arange(batch_size, device=initial_states.device)

    def terminal(rollout):
        if terminal_value_fn is not None:
            return terminal_value_fn(rollout.states[:, :, -1], rollout.times[:, :, -1])
        return rollout.network_values[:, :, -1] if use_network else None

    attacker_plan, defender_plan = nominal_controls, nominal_disturbances
    for _ in range(attacker_config.num_iterations):
        attacker_candidates = sample_control_sequences(
            attacker_plan, num_attacker, attacker_config.noise_std,
            attacker_config.control_lower, attacker_config.control_upper, generator=generator,
            control_hold_steps=attacker_config.control_hold_steps,
            include_axis_candidates=attacker_config.include_axis_candidates)
        defender_candidates = sample_control_sequences(
            defender_plan, num_defender, defender_config.noise_std,
            defender_config.control_lower, defender_config.control_upper, generator=generator,
            control_hold_steps=defender_config.control_hold_steps,
            include_axis_candidates=defender_config.include_axis_candidates)

        table, attacker_violation, defender_violation = [], [], []
        for start in range(0, num_attacker, rows_per_chunk):
            end = min(start + rows_per_chunk, num_attacker)
            rows = end - start
            pair_controls = attacker_candidates[:, start:end, None].expand(
                batch_size, rows, num_defender, horizon_steps, dynamics.control_dim
            ).reshape(batch_size, rows * num_defender, horizon_steps, dynamics.control_dim)
            pair_disturbances = defender_candidates[:, None].expand(
                batch_size, rows, num_defender, horizon_steps, dynamics.disturbance_dim
            ).reshape(batch_size, rows * num_defender, horizon_steps, dynamics.disturbance_dim)
            rollout = rollout_joint_sequences(
                initial_states, initial_times, pair_controls, pair_disturbances, responder, dynamics,
                attacker_config.dt, attacker_config.integration_method, query_mode)
            scores, _ = evaluate_rollouts(dynamics, rollout, terminal(rollout))
            table.append(scores.view(batch_size, rows, num_defender))
            for violations, config in ((attacker_violation, attacker_config), (defender_violation, defender_config)):
                violation = domain_violation(dynamics, config, rollout.states)
                violations.append(torch.zeros_like(scores) if violation is None else violation)
                violations[-1] = violations[-1].view(batch_size, rows, num_defender)

        table = torch.cat(table, dim=1)                                           # [B, Na, Nd]
        attacker_violation = torch.cat(attacker_violation, dim=1).amax(dim=2)     # [B, Na]
        defender_violation = torch.cat(defender_violation, dim=1).amax(dim=1)     # [B, Nd]
        attacker_valid, defender_valid = attacker_violation <= 0, defender_violation <= 0
        attacker_valid = attacker_valid | ~attacker_valid.any(dim=1, keepdim=True)
        defender_valid = defender_valid | ~defender_valid.any(dim=1, keepdim=True)

        attacker_worst = table.masked_fill(~defender_valid[:, None, :], float("-inf")).amax(dim=2)  # [B, Na]
        defender_worst = table.masked_fill(~attacker_valid[:, :, None], float("inf")).amin(dim=1)   # [B, Nd]
        best_attacker = (attacker_worst + _violation_penalty(attacker_violation)).argmin(dim=1)
        best_defender = (defender_worst - _violation_penalty(defender_violation)).argmax(dim=1)
        attacker_plan = attacker_candidates[batch_index, best_attacker]
        defender_plan = defender_candidates[batch_index, best_defender]
        upper_value = attacker_worst[batch_index, best_attacker]
        lower_value = defender_worst[batch_index, best_defender]

    rollout = rollout_joint_sequences(
        initial_states, initial_times, attacker_plan[:, None], defender_plan[:, None], responder, dynamics,
        attacker_config.dt, attacker_config.integration_method, query_mode)
    if responder is not None:
        rollout.network_values[:, :, 0] = responder.query(initial_states, initial_times).values[:, None]
    scores, suffix_values = evaluate_rollouts(dynamics, rollout, terminal(rollout))
    return MPCResult(
        controls=attacker_plan, states=rollout.states[:, 0], defender_controls=defender_plan,
        network_values=rollout.network_values[:, 0], suffix_values=suffix_values[:, 0],
        score=scores[:, 0], all_scores=table.flatten(1),
        upper_value=upper_value, lower_value=lower_value,
    )


def shift_control_sequence(sequence):
    """Receding-horizon warm start: drop the first action, repeat the final action."""
    return torch.cat((sequence[..., 1:, :], sequence[..., -1:, :]), dim=-2)


def closed_loop_rollout(
    initial_states, initial_times, nominal_controls, nominal_disturbances,
    responder: NeuralBangBangController, dynamics,
    attacker_config: MPCConfig, defender_config: MPCConfig, optimized_player="joint",
    replan_every: int = 1,
    generator: Optional[torch.Generator] = None,
    terminal_value_fn: Optional[Callable] = None,
    use_network_terminal_value: bool = False,
    end_on_event: bool = False,
    game_solver: str = "alternating",
) -> MPCResult:
    """Receding-horizon rollout: re-optimise every `replan_every` steps from the state actually reached.

    optimized_player selects who plans with MPC ('attacker', 'defender' or 'joint'); a player that is
    not optimised reacts to the current state with the responder's bang-bang policy at every step.
    The planning horizon shrinks so every plan ends at the same final step, hence replan_every equal
    to the horizon reproduces the corresponding open-loop optimisation exactly.
    With end_on_event, a trajectory stops (its state is frozen) at the first state inside the reach set
    (reach_fn <= 0, e.g. target hit) or the avoid set (avoid_fn <= 0, e.g. capture); its label at that
    state is then max(reach, -avoid) of that state, as if the game ended there.
    game_solver picks the joint planner: 'alternating' (optimize_joint_sequences) or 'maxmin'
    (optimize_maxmin_sequences). With optimized_player='joint' the responder may be None: the
    network is then not used at all and the terminal value is the terminal-set margin.
    Returns the executed trajectory, labelled with the reach-avoid recursion over it.
    """
    if game_solver not in ("alternating", "maxmin"):
        raise ValueError("game_solver must be 'alternating' or 'maxmin'")
    if responder is None and optimized_player != "joint":
        raise ValueError("a responder is required unless optimized_player is 'joint'")
    if optimized_player not in ("attacker", "defender", "joint"):
        raise ValueError("optimized_player must be 'attacker', 'defender', or 'joint'")
    if replan_every < 1:
        raise ValueError("replan_every must be >= 1")
    config = attacker_config if optimized_player != "defender" else defender_config
    horizon_steps, dt = config.horizon_steps, config.dt
    if optimized_player == "joint" and (defender_config.horizon_steps != horizon_steps or defender_config.dt != dt):
        raise ValueError("attacker and defender configs must share dt and horizon")

    batch_size, state_dim = initial_states.shape
    device, dtype = initial_states.device, initial_states.dtype
    current_times = torch.as_tensor(initial_times, dtype=dtype, device=device).reshape(-1).expand(batch_size).clone()
    current_states = initial_states.clone()
    states = torch.zeros(batch_size, horizon_steps + 1, state_dim, device=device, dtype=dtype)
    controls = torch.zeros(batch_size, horizon_steps, dynamics.control_dim, device=device, dtype=dtype)
    disturbances = torch.zeros(batch_size, horizon_steps, dynamics.disturbance_dim, device=device, dtype=dtype)
    times = torch.zeros(batch_size, horizon_steps + 1, device=device, dtype=dtype)
    states[:, 0], times[:, 0] = current_states, current_times

    plan_u, plan_d = nominal_controls, nominal_disturbances
    plan_start = 0
    event_steps = torch.full((batch_size,), horizon_steps, dtype=torch.long, device=device)
    ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for step in range(horizon_steps):
        offset = step - plan_start
        if step % replan_every == 0:
            steps_left = horizon_steps - step
            if offset:   # receding-horizon warm start: drop executed actions, keep the plan length
                if plan_u is not None:
                    plan_u = plan_u[:, offset:offset + steps_left]
                if plan_d is not None:
                    plan_d = plan_d[:, offset:offset + steps_left]
            attacker_now = replace(attacker_config, horizon_steps=steps_left)
            defender_now = replace(defender_config, horizon_steps=steps_left)
            if optimized_player == "joint":
                solver = optimize_maxmin_sequences if game_solver == "maxmin" else optimize_joint_sequences
                result = solver(
                    current_states, current_times, plan_u, plan_d, responder, dynamics,
                    attacker_now, defender_now, generator, terminal_value_fn, use_network_terminal_value)
                plan_u, plan_d = result.controls, result.defender_controls
            elif optimized_player == "attacker":
                plan_u = optimize_control_sequence(
                    current_states, current_times, plan_u, responder, dynamics, attacker_now,
                    generator, terminal_value_fn, use_network_terminal_value).controls
            else:
                plan_d = optimize_disturbance_sequence(
                    current_states, current_times, plan_d, responder, dynamics, defender_now,
                    generator, terminal_value_fn, use_network_terminal_value).defender_controls
            plan_start, offset = step, 0

        if optimized_player == "joint":
            u, d = plan_u[:, offset], plan_d[:, offset]
        else:
            query = responder.query(current_states, current_times)
            u = plan_u[:, offset] if optimized_player == "attacker" else query.controls
            d = plan_d[:, offset] if optimized_player == "defender" else query.disturbances

        if end_on_event:
            newly_ended = ~ended & ((dynamics.reach_fn(current_states) <= 0) | (dynamics.avoid_fn(current_states) <= 0))
            event_steps = torch.where(newly_ended, torch.full_like(event_steps, step), event_steps)
            ended = ended | newly_ended
            u = torch.where(ended.unsqueeze(-1), torch.zeros_like(u), u)
            d = torch.where(ended.unsqueeze(-1), torch.zeros_like(d), d)
        next_states = integrate_step(dynamics, current_states, u, d, dt, config.integration_method).detach()
        active = (current_times > 0) & ~ended
        current_states = torch.where(active.unsqueeze(-1), next_states, current_states)
        current_times = torch.clamp(current_times - dt, min=0.0)
        states[:, step + 1], times[:, step + 1] = current_states, current_times
        controls[:, step], disturbances[:, step] = u, d

    if end_on_event:   # the final state itself may be the first one inside a terminal set
        newly_ended = ~ended & ((dynamics.reach_fn(current_states) <= 0) | (dynamics.avoid_fn(current_states) <= 0))
        ended = ended | newly_ended

    final_values = (torch.zeros(batch_size, device=device, dtype=dtype) if responder is None
                    else responder.query(current_states, current_times).values)
    terminal_values = None
    if terminal_value_fn is not None:
        terminal_values = terminal_value_fn(current_states, current_times)
    elif use_network_terminal_value and responder is not None:
        terminal_values = final_values
    if end_on_event:   # a finished game has no future: its value is its terminal-set margin
        boundary = torch.maximum(dynamics.reach_fn(current_states), -dynamics.avoid_fn(current_states))
        terminal_values = boundary if terminal_values is None else torch.where(ended, boundary, terminal_values)
    suffix_values = reach_avoid_suffix_values(dynamics, states, terminal_values)
    network_values = torch.zeros(batch_size, horizon_steps + 1, device=device, dtype=dtype)
    network_values[:, -1] = final_values
    return MPCResult(
        controls=controls, states=states, defender_controls=disturbances,
        network_values=network_values, suffix_values=suffix_values,
        score=suffix_values[:, 0], all_scores=suffix_values[:, :1],
        event_steps=event_steps if end_on_event else None,
    )
