import math

import torch


class MPCReplayBuffer:
    """Fixed-capacity FIFO store of (time-to-go, state, BRAT value) labels, real units, on CPU."""

    def __init__(self, state_dim: int, capacity: int):
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.state_dim = state_dim
        self.capacity = capacity
        self.times = torch.zeros(0)
        self.states = torch.zeros(0, state_dim)
        self.values = torch.zeros(0)

    def __len__(self):
        return self.times.shape[0]

    def add(self, times: torch.Tensor, states: torch.Tensor, values: torch.Tensor):
        times = times.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        states = states.detach().to(dtype=torch.float32, device="cpu").reshape(-1, self.state_dim)
        values = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)

        if not (times.shape[0] == states.shape[0] == values.shape[0]):
            raise ValueError("times, states, and values must have matching leading dimensions")
        if not torch.all(torch.isfinite(times)) or not torch.all(torch.isfinite(states)) or not torch.all(torch.isfinite(values)):
            raise ValueError("times, states, and values must be finite")

        self.times = torch.cat((self.times, times), dim=0)[-self.capacity:]
        self.states = torch.cat((self.states, states), dim=0)[-self.capacity:]
        self.values = torch.cat((self.values, values), dim=0)[-self.capacity:]

    def sample(self, batch_size: int, device, generator: torch.Generator = None):
        if len(self) == 0:
            raise RuntimeError("cannot sample from an empty MPCReplayBuffer")
        indices = torch.randint(len(self), (batch_size,), generator=generator)
        return (
            self.times[indices].to(device),
            self.states[indices].to(device),
            self.values[indices].to(device),
        )

    def sample_up_to_time(self, batch_size: int, max_time: float, generator: torch.Generator = None):
        """Up to batch_size distinct labels with time-to-go <= max_time, drawn without replacement
        (fewer if fewer are eligible); None if there are none. CPU tensors."""
        eligible = torch.nonzero(self.times <= max_time + 1e-6).squeeze(-1)
        if eligible.numel() == 0:
            return None
        indices = eligible[torch.randperm(eligible.numel(), generator=generator)[:batch_size]]
        return self.times[indices], self.states[indices], self.values[indices]

    def state_dict(self):
        return {"times": self.times, "states": self.states, "values": self.values}

    def load_state_dict(self, state):
        self.times = state["times"]
        self.states = state["states"]
        self.values = state["values"]


def sample_mpc_initial_states(
        dynamics, num_samples, distribution='uniform',
        defender_position_std=0.5, attacker_boundary_std=0.2,
        attacker_velocity='uniform', attacker_velocity_spread_deg=60.0,
        attacker_speed_max=None):
    """Real-unit initial states for MPC label generation.

    attacker_velocity='inward' points the attacker velocity at the target (origin)
    rotated by a uniform angle in +-attacker_velocity_spread_deg, with a uniform speed
    in [0, attacker_speed_max]; 'uniform' samples each velocity component independently.
    """
    if distribution == 'uniform':
        model_states = torch.zeros(num_samples, dynamics.state_dim).uniform_(-1, 1)
        return dynamics.input_to_coord(
            torch.cat((torch.zeros(num_samples, 1), model_states), dim=1)
        )[:, 1:]
    if distribution != 'interception':
        raise ValueError("distribution must be 'uniform' or 'interception'")
    if dynamics.state_dim != 8 or not all(
            hasattr(dynamics, name) for name in ('target_R', 'capture_R')):
        raise ValueError("interception sampling requires the 8D interception dynamics")
    if defender_position_std <= 0 or attacker_boundary_std <= 0:
        raise ValueError('position standard deviations must be positive')

    state_mean = dynamics.state_mean.to(dtype=torch.float32)
    state_var = dynamics.state_var.to(dtype=torch.float32)
    lower = state_mean - state_var
    upper = state_mean + state_var
    states = state_mean.unsqueeze(0).expand(num_samples, -1).clone()

    side = torch.randint(4, (num_samples,))
    inward_distance = torch.abs(torch.randn(num_samples)) * attacker_boundary_std
    horizontal_side = side < 2
    states[:, 0] = torch.empty(num_samples).uniform_(lower[0].item(), upper[0].item())
    states[:, 2] = torch.empty(num_samples).uniform_(lower[2].item(), upper[2].item())
    states[horizontal_side, 0] = torch.where(
        side[horizontal_side] == 0,
        lower[0] + inward_distance[horizontal_side],
        upper[0] - inward_distance[horizontal_side],
    ).clamp(lower[0], upper[0])
    states[~horizontal_side, 2] = torch.where(
        side[~horizontal_side] == 2,
        lower[2] + inward_distance[~horizontal_side],
        upper[2] - inward_distance[~horizontal_side],
    ).clamp(lower[2], upper[2])

    exclusion_radius = getattr(dynamics, 'defender_exclusion_R', dynamics.target_R + dynamics.capture_R)
    accepted_positions = []
    accepted_count = 0
    while accepted_count < num_samples:
        candidates = torch.randn(max(2 * (num_samples - accepted_count), 16), 2) * defender_position_std
        inside_domain = (
            (candidates[:, 0] >= lower[4]) & (candidates[:, 0] <= upper[4]) &
            (candidates[:, 1] >= lower[6]) & (candidates[:, 1] <= upper[6]))
        outside_exclusion = torch.linalg.vector_norm(candidates, dim=-1) > exclusion_radius
        accepted = candidates[inside_domain & outside_exclusion]
        accepted_positions.append(accepted)
        accepted_count += accepted.shape[0]
    defender_positions = torch.cat(accepted_positions, dim=0)[:num_samples]
    states[:, 4] = defender_positions[:, 0]
    states[:, 6] = defender_positions[:, 1]
    states[:, 1] = torch.empty(num_samples).uniform_(lower[1].item(), upper[1].item())
    states[:, 3] = torch.empty(num_samples).uniform_(lower[3].item(), upper[3].item())
    states[:, [5, 7]] = 0.0
    if attacker_velocity == 'inward':
        speed_limit = min(upper[1].item(), upper[3].item())
        if getattr(dynamics, 'vel_max_a', None) is not None:
            speed_limit = min(speed_limit, dynamics.vel_max_a)
        speed_max = speed_limit if attacker_speed_max is None else attacker_speed_max
        heading = torch.atan2(-states[:, 2], -states[:, 0]) + torch.empty(num_samples).uniform_(
            -math.radians(attacker_velocity_spread_deg), math.radians(attacker_velocity_spread_deg))
        speed = torch.empty(num_samples).uniform_(0.0, speed_max)
        states[:, 1] = (speed * torch.cos(heading)).clamp(lower[1], upper[1])
        states[:, 3] = (speed * torch.sin(heading)).clamp(lower[3], upper[3])
    elif attacker_velocity != 'uniform':
        raise ValueError("attacker_velocity must be 'uniform' or 'inward'")
    if hasattr(dynamics, 'limit_state'):   # respect speed limits in the initial states too
        states = dynamics.limit_state(states)
    return states


def sample_mpc_initial_times(dataset, num_samples, distribution='uniform'):
    """Time-to-go for MPC initial states: 'uniform' follows the training curriculum, 'tmax' starts every rollout at tMax."""
    if distribution == 'uniform':
        return dataset._sample_times(num_samples).squeeze(-1)
    if distribution == 'current_max':   # every game starts at the current curriculum maximum
        return torch.full((num_samples,), float(dataset._current_t_max()))
    if distribution == 'tmax':
        if dataset._current_t_max() < dataset.tMax:
            raise ValueError(
                'mpc_time_distribution=tmax requires the time curriculum to have reached tMax '
                '(set mpc_start_epoch after pretraining and counter_end)')
        return torch.full((num_samples,), float(dataset.tMax))
    raise ValueError("distribution must be 'uniform', 'current_max' or 'tmax'")


def mpc_domain_constraint(dynamics, player, mode='position', defender_keep_out=True):
    """MPCConfig kwargs restricting `player`'s own drone to the training domain (state_mean +- state_var).

    mode: 'none', 'position' (own x/y positions) or 'state' (own positions and velocities).
    defender_keep_out additionally rejects defender plans entering the defender exclusion zone.
    Only defined for the 8D interception state [px_a, vx_a, py_a, vy_a, px_d, vx_d, py_d, vy_d].
    """
    keep_out = {}
    if defender_keep_out and player == 'defender' and hasattr(dynamics, 'defender_exclusion_R'):
        keep_out = dict(keep_out_dims=(4, 6), keep_out_radius=float(dynamics.defender_exclusion_R))
    if mode == 'none':
        return keep_out
    if mode not in ('position', 'state'):
        raise ValueError("mode must be 'none', 'position' or 'state'")
    if dynamics.state_dim != 8:
        raise ValueError('domain constraint is only defined for the 8D interception dynamics')
    offset = {'attacker': 0, 'defender': 4}[player]
    dims = tuple(offset + i for i in ((0, 2) if mode == 'position' else (0, 1, 2, 3)))
    mean = dynamics.state_mean.to(dtype=torch.float32)
    var = dynamics.state_var.to(dtype=torch.float32)
    return dict(domain_dims=dims, domain_lower=(mean - var)[list(dims)], domain_upper=(mean + var)[list(dims)],
                **keep_out)


def in_domain_mask(dynamics, states):
    """[...] True where the whole state lies inside the training domain state_mean +- state_var."""
    mean = dynamics.state_mean.to(dtype=states.dtype, device=states.device)
    var = dynamics.state_var.to(dtype=states.dtype, device=states.device)
    return ((states - mean).abs() <= var + 1e-6).all(dim=-1)


def mpc_label_times(dynamics, result, initial_times, dt):
    """Time-to-go attached to every state of MPC trajectories [B, H+1].

    A game that ended at an event (capture, target hit, breach) is labelled as a game that ends exactly
    at that event: the state k steps before the event gets time-to-go k * dt, so the event state itself
    gets 0. Its label (the reach-avoid score with the event state's terminal margin) is then the value
    of exactly that game, which is what the network's V(t, x) means. Games that ran out of time keep
    their real time-to-go.
    """
    steps = result.states.shape[1]
    step_index = torch.arange(steps, device=result.states.device)[None]
    label_times = torch.clamp(initial_times.reshape(-1, 1) - step_index * dt, min=0.0)
    event_steps = getattr(result, 'event_steps', None)
    if event_steps is None:
        return label_times
    event_states = result.states[torch.arange(result.states.shape[0], device=result.states.device), event_steps]
    ended_by_event = (dynamics.reach_fn(event_states) <= 0) | (dynamics.avoid_fn(event_states) <= 0)
    to_event = torch.clamp((event_steps[:, None] - step_index) * dt, min=0.0)
    return torch.where(ended_by_event[:, None], to_event, label_times)


def outcome_metrics(predicted, label):
    """How well predicted values match game outcomes. Sign <= 0 means attacker wins.

    Returns accuracy on attacker-win games, on defender-win games, their mean (balanced accuracy,
    which is not fooled by one outcome being much more common), the share of attacker wins, the mean
    signed error (predicted - label; > 0 means too favourable to the defender) and the mean |error|.
    """
    predicted, label = predicted.reshape(-1).float(), label.reshape(-1).float()
    attacker_wins, predicted_attacker_wins = label <= 0, predicted <= 0
    metrics = {'count': float(label.numel()), 'attacker_win_share': attacker_wins.float().mean().item(),
               'mean_error': (predicted - label).mean().item(),
               'mean_abs_error': (predicted - label).abs().mean().item()}
    rates = []
    if attacker_wins.any():
        metrics['attacker_win_accuracy'] = predicted_attacker_wins[attacker_wins].float().mean().item()
        rates.append(metrics['attacker_win_accuracy'])
    if (~attacker_wins).any():
        metrics['defender_win_accuracy'] = (~predicted_attacker_wins[~attacker_wins]).float().mean().item()
        rates.append(metrics['defender_win_accuracy'])
    metrics['balanced_accuracy'] = sum(rates) / len(rates)
    return metrics


TIME_BINS = (0.0, 0.5, 1.0, 1.5, float('inf'))


def abs_error_by_time(predicted, label, times, bins=TIME_BINS):
    """Mean |predicted - label| per time-to-go bin, keyed like 't0.0-0.5'; empty bins are skipped."""
    errors = (predicted.reshape(-1) - label.reshape(-1)).abs()
    times = times.reshape(-1)
    result = {}
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (times >= low) & (times < high)
        if mask.any():
            result['t%.1f-%s' % (low, 'max' if high == float('inf') else '%.1f' % high)] = errors[mask].mean().item()
    return result
