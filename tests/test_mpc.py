import torch

from controllers.bang_bang import BangBangQuery
from controllers.mpc import (
    MPCConfig,
    integrate_step,
    optimize_control_sequence,
    optimize_disturbance_sequence,
    optimize_joint_sequences,
    reach_avoid_suffix_values,
    rollout_control_sequences,
    rollout_disturbance_sequences,
    rollout_joint_sequences,
    sample_control_sequences,
    shift_control_sequence,
)
from dynamics.dynamics import CrazyflieInterception


def make_dynamics():
    return CrazyflieInterception(target_R=0.25, capture_R=0.2, accel_max_a=5.0, accel_max_d=7.0)


class ZeroDisturbanceResponder:
    """Fake responder: defender always applies zero acceleration; values are unused by tests."""

    def query(self, states, times):
        batch = states.shape[0]
        return BangBangQuery(
            values=torch.zeros(batch),
            gradients=torch.zeros(batch, states.shape[-1]),
            controls=torch.zeros(batch, 2),
            disturbances=torch.zeros(batch, 2),
        )

    def __call__(self, states, times):
        return self.query(states, times)


# ---------------------------------------------------------------------------
# sample_control_sequences
# ---------------------------------------------------------------------------

def test_sample_control_sequences_shape_bounds_and_nominal():
    nominal = torch.zeros(2, 5, 2)
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    generator = torch.Generator().manual_seed(0)

    samples = sample_control_sequences(nominal, num_samples=8, noise_std=5.0, lower=lower, upper=upper, generator=generator)

    assert samples.shape == (2, 8, 5, 2)
    assert torch.all(samples >= lower)
    assert torch.all(samples <= upper)
    assert torch.equal(samples[:, 0], nominal)


def test_sample_control_sequences_includes_constant_axis_extremes():
    nominal = torch.zeros(1, 5, 2)
    samples = sample_control_sequences(
        nominal, num_samples=5, noise_std=1.0,
        lower=torch.tensor([-5.0, -5.0]), upper=torch.tensor([5.0, 5.0]),
        include_axis_candidates=True,
    )

    assert torch.equal(samples[0, 0], nominal[0])
    assert torch.equal(samples[0, 1], torch.tensor([[-5.0, 0.0]]).expand(5, 2))
    assert torch.equal(samples[0, 2], torch.tensor([[5.0, 0.0]]).expand(5, 2))
    assert torch.equal(samples[0, 3], torch.tensor([[0.0, -5.0]]).expand(5, 2))
    assert torch.equal(samples[0, 4], torch.tensor([[0.0, 5.0]]).expand(5, 2))


def test_sample_control_sequences_zero_noise_returns_nominal():
    nominal = torch.rand(1, 3, 2) * 2 - 1
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])

    samples = sample_control_sequences(nominal, num_samples=4, noise_std=0.0, lower=lower, upper=upper)

    for candidate in range(4):
        assert torch.allclose(samples[:, candidate], nominal)


def test_sample_control_sequences_deterministic_with_seed():
    nominal = torch.zeros(1, 2, 2)
    lower = torch.tensor([-2.0, -2.0])
    upper = torch.tensor([2.0, 2.0])

    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    samples_a = sample_control_sequences(nominal, 4, 1.0, lower, upper, generator=gen_a)
    samples_b = sample_control_sequences(nominal, 4, 1.0, lower, upper, generator=gen_b)

    assert torch.equal(samples_a, samples_b)


def test_sample_control_sequences_holds_noise_within_blocks():
    nominal = torch.zeros(1, 7, 2)
    samples = sample_control_sequences(
        nominal, 4, 1.0, torch.tensor([-2.0, -2.0]), torch.tensor([2.0, 2.0]),
        generator=torch.Generator().manual_seed(3), control_hold_steps=3,
    )

    assert torch.equal(samples[:, 1:, 0], samples[:, 1:, 1])
    assert torch.equal(samples[:, 1:, 1], samples[:, 1:, 2])
    assert torch.equal(samples[:, 1:, 3], samples[:, 1:, 4])
    assert torch.equal(samples[:, 1:, 4], samples[:, 1:, 5])


# ---------------------------------------------------------------------------
# integrate_step
# ---------------------------------------------------------------------------

def test_integrate_step_euler_matches_manual_update():
    dynamics = make_dynamics()
    state = torch.zeros(1, 8)
    control = torch.tensor([[1.0, 2.0]])
    disturbance = torch.tensor([[0.0, 0.0]])
    dt = 0.1

    next_state = integrate_step(dynamics, state, control, disturbance, dt, "euler")
    expected = state + dt * dynamics.dsdt(state, control, disturbance)

    assert torch.allclose(next_state, expected)


def test_integrate_step_rk4_matches_analytical_constant_acceleration():
    dynamics = make_dynamics()
    state = torch.zeros(1, 8)
    control = torch.tensor([[2.0, -1.0]])
    disturbance = torch.tensor([[0.5, 0.5]])
    dt = 0.05

    next_state = integrate_step(dynamics, state, control, disturbance, dt, "rk4")

    # Analytical double-integrator solution for constant acceleration: x' = x + v*dt + 0.5*a*dt^2, v' = v + a*dt
    expected_px_a = 0.5 * control[0, 0] * dt ** 2
    expected_vx_a = control[0, 0] * dt
    assert torch.isclose(next_state[0, 0], expected_px_a, atol=1e-6)
    assert torch.isclose(next_state[0, 1], expected_vx_a, atol=1e-6)


def test_integrate_step_unknown_method_raises():
    dynamics = make_dynamics()
    state = torch.zeros(1, 8)
    control = torch.zeros(1, 2)
    disturbance = torch.zeros(1, 2)
    try:
        integrate_step(dynamics, state, control, disturbance, 0.1, "bogus")
    except ValueError:
        return
    assert False, "expected ValueError for unknown integration method"


# ---------------------------------------------------------------------------
# rollout_control_sequences
# ---------------------------------------------------------------------------

def test_rollout_control_sequences_shapes():
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()

    initial_states = torch.zeros(2, 8)
    initial_times = torch.tensor([0.5, 0.5])
    attacker_controls = torch.zeros(2, 3, 4, 2)

    rollout = rollout_control_sequences(initial_states, initial_times, attacker_controls, responder, dynamics, dt=0.1)

    assert rollout.states.shape == (2, 3, 5, 8)
    assert rollout.defender_controls.shape == (2, 3, 4, 2)
    assert rollout.network_values.shape == (2, 3, 5)
    assert rollout.times.shape == (2, 3, 5)


def test_rollout_control_sequences_freezes_after_time_expires():
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()

    initial_states = torch.zeros(1, 8)
    initial_times = torch.tensor([0.05])  # expires after one 0.05s step
    attacker_controls = torch.ones(1, 1, 4, 2) * 5.0

    rollout = rollout_control_sequences(initial_states, initial_times, attacker_controls, responder, dynamics, dt=0.05)

    # times should hit zero and stay there
    assert torch.allclose(rollout.times[:, :, 1], torch.zeros(1, 1))
    assert torch.allclose(rollout.times[:, :, -1], torch.zeros(1, 1))
    # state should not change after freezing (steps 2..4 identical to step 1)
    assert torch.allclose(rollout.states[:, :, 1], rollout.states[:, :, -1])


def test_rollout_joint_sequences_uses_both_prescribed_sequences():
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()
    initial_states = torch.zeros(1, 8)
    attacker_controls = torch.tensor([[[[2.0, -1.0], [2.0, -1.0]]]])
    defender_controls = torch.tensor([[[[-3.0, 4.0], [-3.0, 4.0]]]])

    rollout = rollout_joint_sequences(
        initial_states, torch.tensor([0.2]), attacker_controls, defender_controls,
        responder, dynamics, dt=0.1,
    )

    assert torch.equal(rollout.attacker_controls, attacker_controls)
    assert torch.equal(rollout.defender_controls, defender_controls)
    assert torch.allclose(rollout.states[0, 0, 1, [1, 3]], torch.tensor([0.2, -0.1]))
    assert torch.allclose(rollout.states[0, 0, 1, [5, 7]], torch.tensor([-0.3, 0.4]))


# ---------------------------------------------------------------------------
# reach_avoid_suffix_values
# ---------------------------------------------------------------------------

def _brute_force_suffix_values(dynamics, states):
    reach = dynamics.reach_fn(states)
    failure = -dynamics.avoid_fn(states)
    horizon = states.shape[-2]
    result = torch.empty_like(reach)
    for start in range(horizon):
        best = None
        for end in range(start, horizon):
            candidate = torch.maximum(reach[..., end], torch.amax(failure[..., start:end + 1], dim=-1))
            best = candidate if best is None else torch.minimum(best, candidate)
        result[..., start] = best
    return result


def test_reach_avoid_suffix_values_matches_brute_force():
    dynamics = make_dynamics()
    torch.manual_seed(0)
    states = torch.randn(1, 6, 8) * 0.5

    fast = reach_avoid_suffix_values(dynamics, states)
    slow = _brute_force_suffix_values(dynamics, states)

    assert torch.allclose(fast, slow, atol=1e-5)


def test_reach_avoid_score_target_reached_before_capture():
    dynamics = make_dynamics()
    # attacker starts far, ends at origin (reached); defender always far from attacker and target.
    states = torch.zeros(1, 3, 8)
    states[0, 0, 0] = 2.0   # px_a far
    states[0, 1, 0] = 1.0
    states[0, 2, 0] = 0.0   # px_a at target
    states[0, :, 4] = 5.0   # px_d always far from origin and attacker

    scores, _ = (lambda r: (reach_avoid_suffix_values(dynamics, r)[..., 0], None))(states)
    assert scores.item() <= 0.0  # success: reach margin <= 0 achieved without prior capture


def test_reach_avoid_score_captured_before_target():
    dynamics = make_dynamics()
    states = torch.zeros(1, 2, 8)
    # step 0: attacker far from target, close to defender (captured)
    states[0, 0, 0] = 2.0
    states[0, 0, 4] = 2.0 + 0.05  # within capture_R=0.2 of attacker
    # step 1: attacker reaches target, but capture already occurred
    states[0, 1, 0] = 0.0
    states[0, 1, 4] = 5.0

    score = reach_avoid_suffix_values(dynamics, states)[..., 0]
    assert score.item() > 0.0  # failure: captured before reaching target


# ---------------------------------------------------------------------------
# optimize_control_sequence
# ---------------------------------------------------------------------------

def test_optimize_control_sequence_score_improves_or_stays(monkeypatch=None):
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()

    initial_states = torch.zeros(1, 8)
    initial_states[0, 0] = 1.5  # attacker starts away from target
    initial_times = torch.tensor([0.5])
    nominal = torch.zeros(1, 10, 2)

    config = MPCConfig(
        dt=0.05, horizon_steps=10, num_samples=32, num_iterations=3, noise_std=5.0,
        control_lower=torch.tensor([-5.0, -5.0]), control_upper=torch.tensor([5.0, 5.0]),
    )
    generator = torch.Generator().manual_seed(1)

    result = optimize_control_sequence(initial_states, initial_times, nominal, responder, dynamics, config, generator=generator)

    # driving toward the origin should score no worse than doing nothing
    zero_rollout = rollout_control_sequences(initial_states, initial_times, torch.zeros(1, 1, 10, 2), responder, dynamics, dt=0.05)
    zero_score = reach_avoid_suffix_values(dynamics, zero_rollout.states)[..., 0]

    assert result.score.item() <= zero_score.item() + 1e-6


def test_optimize_control_sequence_candidate_zero_preserves_nominal_quality():
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()

    initial_states = torch.zeros(1, 8)
    initial_times = torch.tensor([0.2])
    nominal = torch.zeros(1, 4, 2)

    config = MPCConfig(
        dt=0.05, horizon_steps=4, num_samples=1, num_iterations=1, noise_std=0.0,
        control_lower=torch.tensor([-5.0, -5.0]), control_upper=torch.tensor([5.0, 5.0]),
    )

    result = optimize_control_sequence(initial_states, initial_times, nominal, responder, dynamics, config)
    assert torch.equal(result.controls, nominal)


def test_optimize_control_sequence_considers_constant_axis_extremes():
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()
    initial_states = torch.tensor([[-1.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7, 0.0]])
    initial_times = torch.tensor([1.0])
    nominal = torch.zeros(1, 50, 2)
    config = MPCConfig(
        dt=0.02, horizon_steps=50, num_samples=5, num_iterations=1, noise_std=0.0,
        control_lower=torch.tensor([-5.0, -5.0]), control_upper=torch.tensor([5.0, 5.0]),
        control_hold_steps=10,
    )

    result = optimize_control_sequence(
        initial_states, initial_times, nominal, responder, dynamics, config)

    assert result.controls[0, 0, 0] > 0.0
    assert result.score.item() < 0.0


def test_optimize_disturbance_sequence_maximizes_relative_to_nominal():
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()
    initial_states = torch.zeros(1, 8)
    initial_states[0, 0] = 1.0
    initial_states[0, 4] = 0.0
    initial_states[0, 6] = 1.0
    initial_times = torch.tensor([0.3])
    nominal = torch.zeros(1, 6, dynamics.disturbance_dim)
    config = MPCConfig(
        dt=0.05, horizon_steps=6, num_samples=32, num_iterations=2, noise_std=4.0,
        control_lower=torch.tensor([-7.0, -7.0]), control_upper=torch.tensor([7.0, 7.0]),
    )

    nominal_rollout = rollout_disturbance_sequences(
        initial_states, initial_times, nominal.unsqueeze(1), responder, dynamics, dt=0.05)
    nominal_score = reach_avoid_suffix_values(dynamics, nominal_rollout.states)[..., 0]
    result = optimize_disturbance_sequence(
        initial_states, initial_times, nominal, responder, dynamics, config,
        generator=torch.Generator().manual_seed(7),
    )

    assert result.score.item() >= nominal_score.item() - 1e-6
    assert result.controls.shape == (1, 6, dynamics.control_dim)
    assert result.defender_controls.shape == (1, 6, dynamics.disturbance_dim)


def test_optimize_joint_sequences_preserves_nominals_with_one_candidate():
    dynamics = make_dynamics()
    responder = ZeroDisturbanceResponder()
    initial_states = torch.zeros(1, 8)
    initial_times = torch.tensor([0.2])
    nominal_controls = torch.ones(1, 4, 2)
    nominal_disturbances = -torch.ones(1, 4, 2)
    attacker_config = MPCConfig(
        dt=0.05, horizon_steps=4, num_samples=1, num_iterations=2, noise_std=0.0,
        control_lower=torch.tensor([-5.0, -5.0]), control_upper=torch.tensor([5.0, 5.0]),
    )
    defender_config = MPCConfig(
        dt=0.05, horizon_steps=4, num_samples=1, num_iterations=2, noise_std=0.0,
        control_lower=torch.tensor([-7.0, -7.0]), control_upper=torch.tensor([7.0, 7.0]),
    )

    result = optimize_joint_sequences(
        initial_states, initial_times, nominal_controls, nominal_disturbances,
        responder, dynamics, attacker_config, defender_config,
    )

    assert torch.equal(result.controls, nominal_controls)
    assert torch.equal(result.defender_controls, nominal_disturbances)


# ---------------------------------------------------------------------------
# shift_control_sequence
# ---------------------------------------------------------------------------

def test_shift_control_sequence():
    sequence = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    shifted = shift_control_sequence(sequence)
    expected = torch.tensor([[[2.0, 2.0], [3.0, 3.0], [3.0, 3.0]]])
    assert torch.equal(shifted, expected)
