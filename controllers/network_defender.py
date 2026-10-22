"""The trained value network as a defender controller (CrazyflieInterception state layout).

NetworkDefender wraps NeuralBangBangController and has the same query(states, times) interface, so it can also
stand in for the network inside MPC rollouts. Only the defender's action (query.disturbances) is changed:
  policy 'gradient'   bang-bang from the value's slope with respect to the defender's velocity
  policy 'lookahead'  among 9 accelerations (4 box corners, 4 axis directions, none), the one whose predicted
                      state after lookahead_time has the highest V (the attacker keeps its velocity meanwhile)
  time_to_go          query V at this fixed time-to-go (the "unlimited time" value) instead of the given times
  safety_filter       an acceleration is allowed only if, after applying it for one step and then braking fully,
                      the defender stops outside its keep-out zone (and, with the box rule, inside the box);
                      an unsafe choice is replaced by the best safe one of the 9 candidates (by V for lookahead,
                      by the velocity slope for gradient), or by full braking if none is safe
"""

import torch

from controllers.bang_bang import NeuralBangBangController
from controllers.mpc import integrate_step

SAFETY_MARGIN = 0.005   # m kept clear of the keep-out circle and the box walls


class NetworkDefender:
    def __init__(self, model, dynamics, device, policy='gradient', time_to_go=None, lookahead_time=0.2, dt=0.02,
                 safety_filter=True):
        if policy not in ('gradient', 'lookahead'):
            raise ValueError("policy must be 'gradient' or 'lookahead'")
        self.model, self.dynamics, self.device = model, dynamics, torch.device(device)
        self.inner = NeuralBangBangController(model=model, dynamics=dynamics, device=device)
        self.policy, self.time_to_go, self.dt, self.safety_filter = policy, time_to_go, dt, safety_filter
        self.steps = max(int(round(lookahead_time / dt)), 1)
        self.step_dt = lookahead_time / self.steps
        a = dynamics.accel_max_d
        self.candidates = torch.tensor([[a, a], [a, -a], [-a, a], [-a, -a], [a, 0.0], [-a, 0.0], [0.0, a],
                                        [0.0, -a], [0.0, 0.0]], dtype=torch.float32, device=self.device)
        self.safety_stats = {'steps': 0, 'changed': 0}

    def _times(self, times, count):
        if self.time_to_go is not None:
            return torch.full((count,), float(self.time_to_go), device=self.device)
        return torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(-1).expand(count)

    def values(self, states, times):
        """V at (times or the fixed time-to-go, states) without gradients, [B]."""
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device).reshape(-1, self.dynamics.state_dim)
        times = self._times(times, states.shape[0])
        coords = self.dynamics.coord_to_input(torch.cat((times.unsqueeze(-1), states), dim=-1))
        with torch.no_grad():
            results = self.model({'coords': coords})
            return self.dynamics.io_to_value(results['model_in'], results['model_out'].squeeze(-1))

    def query(self, states, times):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device).reshape(-1, self.dynamics.state_dim)
        batch = states.shape[0]
        times = self._times(times, batch)
        query = self.inner.query(states, times)
        candidates = self.candidates[None].expand(batch, -1, -1)
        if self.policy == 'lookahead':
            count = self.candidates.shape[0]
            predicted = states.repeat_interleave(count, dim=0)
            defender = self.candidates.repeat(batch, 1)
            attacker = torch.zeros_like(defender)                 # the attacker keeps its current velocity
            for _ in range(self.steps):
                predicted = integrate_step(self.dynamics, predicted, attacker, defender, self.step_dt, 'euler')
            value_times = (times.repeat_interleave(count) if self.time_to_go is not None
                           else torch.clamp(times.repeat_interleave(count) - self.steps * self.step_dt, min=0.0))
            scores = self.values(predicted, value_times).view(batch, count)
            chosen = self.candidates[scores.argmax(dim=1)]
        else:
            scores = (candidates * query.gradients[:, None, [5, 7]]).sum(-1)
            chosen = query.disturbances
        if self.safety_filter:
            chosen = self.choose_safe(states, chosen, candidates, scores)
        query.disturbances = chosen
        return query

    def braking_acceleration(self, velocity):
        """[..., 2] full braking: against the velocity, scaled so the larger axis is at accel_max_d."""
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        direction = velocity / speed.clamp_min(1e-9)
        scale = self.dynamics.accel_max_d / torch.amax(direction.abs(), dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.where(speed > 1e-6, -direction * scale, torch.zeros_like(velocity))

    def safe(self, states, accelerations):
        """[B, K] True where accelerating the defender with accelerations [B, K, 2] for one step and then braking
        fully keeps it out of its keep-out zone (and inside the box when the box rule is on)."""
        dynamics, dt = self.dynamics, self.dt
        position = states[:, None, [4, 6]]
        velocity = states[:, None, [5, 7]].expand_as(accelerations)
        position = position + velocity * dt + 0.5 * accelerations * dt ** 2          # after one step
        velocity = velocity + accelerations * dt
        if dynamics.vel_max_d is not None:
            speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
            velocity = velocity * torch.clamp(float(dynamics.vel_max_d) / speed.clamp_min(1e-9), max=1.0)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        direction = velocity / speed.clamp_min(1e-9)[..., None]
        deceleration = torch.linalg.vector_norm(self.braking_acceleration(velocity), dim=-1).clamp_min(1e-9)
        stop = speed ** 2 / (2 * deceleration)                                          # braking distance
        # the braking path is the straight segment position -> position + direction * stop
        along = torch.minimum(torch.clamp(-(position * direction).sum(-1), min=0.0), stop)
        closest = torch.linalg.vector_norm(position + direction * along[..., None], dim=-1)
        safe = closest > dynamics.defender_exclusion_R + SAFETY_MARGIN
        if getattr(dynamics, 'box_loses', False):
            end = position + direction * stop[..., None]
            limit = dynamics.pos_range - SAFETY_MARGIN
            safe = safe & (position.abs().amax(-1) < limit) & (end.abs().amax(-1) < limit)
        return safe

    def choose_safe(self, states, chosen, candidates, scores):
        """Keep `chosen` [B, 2] where it is safe; elsewhere the safe candidate [B, K, 2] with the highest score
        [B, K], or full braking if none is safe."""
        brake = self.braking_acceleration(states[:, [5, 7]])
        options = torch.cat((candidates, brake[:, None], chosen[:, None]), dim=1)
        safe = self.safe(states, options)
        scores = torch.cat((scores, torch.full_like(scores[:, :1], -1e9), torch.full_like(scores[:, :1], -2e9)), dim=1)
        pick = torch.where(safe, scores, torch.full_like(scores, -float('inf'))).argmax(dim=1)
        pick = torch.where(safe[:, :-2].any(dim=1), pick, torch.full_like(pick, candidates.shape[1]))   # brake
        keep = safe[:, -1]
        self.safety_stats['steps'] += states.shape[0]
        self.safety_stats['changed'] += int((~keep).sum())
        return torch.where(keep[:, None], chosen, options[torch.arange(states.shape[0], device=states.device), pick])

    def __getattr__(self, name):
        return getattr(self.inner, name)
