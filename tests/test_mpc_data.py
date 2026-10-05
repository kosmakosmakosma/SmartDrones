import torch
from types import SimpleNamespace

from experiments import experiments
from dynamics.dynamics import CrazyflieInterception
from utils.mpc_data import MPCReplayBuffer


def test_add_and_len():
    buffer = MPCReplayBuffer(state_dim=8, capacity=100)
    buffer.add(torch.zeros(5), torch.zeros(5, 8), torch.zeros(5))
    assert len(buffer) == 5


def test_fifo_capacity_eviction():
    buffer = MPCReplayBuffer(state_dim=1, capacity=10)
    for batch in range(3):
        buffer.add(torch.full((5,), float(batch)), torch.full((5, 1), float(batch)), torch.full((5,), float(batch)))
    assert len(buffer) == 10
    # oldest batch (0) should have been partially evicted; newest batch (2) fully present
    assert torch.all(buffer.values[-5:] == 2.0)


def test_sample_returns_requested_batch_size_on_device():
    buffer = MPCReplayBuffer(state_dim=8, capacity=50)
    buffer.add(torch.rand(20), torch.rand(20, 8), torch.rand(20))
    times, states, values = buffer.sample(6, device="cpu")
    assert times.shape == (6,)
    assert states.shape == (6, 8)
    assert values.shape == (6,)


def test_sample_empty_buffer_raises():
    buffer = MPCReplayBuffer(state_dim=8, capacity=10)
    try:
        buffer.sample(1, device="cpu")
    except RuntimeError:
        return
    assert False, "expected RuntimeError when sampling an empty buffer"


def test_add_rejects_non_finite_values():
    buffer = MPCReplayBuffer(state_dim=2, capacity=10)
    try:
        buffer.add(torch.tensor([float("nan")]), torch.zeros(1, 2), torch.zeros(1))
    except ValueError:
        return
    assert False, "expected ValueError for non-finite input"


def test_state_dict_round_trip():
    buffer = MPCReplayBuffer(state_dim=4, capacity=20)
    buffer.add(torch.rand(7), torch.rand(7, 4), torch.rand(7))

    restored = MPCReplayBuffer(state_dim=4, capacity=20)
    restored.load_state_dict(buffer.state_dict())

    assert len(restored) == len(buffer)
    assert torch.equal(restored.times, buffer.times)
    assert torch.equal(restored.states, buffer.states)
    assert torch.equal(restored.values, buffer.values)


def test_joint_mpc_refresh_adds_joint_trajectory_labels(monkeypatch, tmp_path):
    class DynamicsStub:
        state_dim = 2
        control_dim = 1
        disturbance_dim = 1

        def input_to_coord(self, coordinates):
            return coordinates

        def reach_fn(self, states):
            return states[..., 0]

        def avoid_fn(self, states):
            return states[..., 1]

        def optimal_control(self, states, gradients):
            return torch.zeros(states.shape[0], 1)

        def optimal_disturbance(self, states, gradients):
            return torch.zeros(states.shape[0], 1)

    class DatasetStub:
        dynamics = DynamicsStub()

        def _sample_times(self, count):
            return torch.ones(count, 1)

    def fake_joint_optimizer(
            states, times, nominal_controls, nominal_disturbances,
            responder, dynamics, attacker_config, defender_config, **kwargs):
        assert torch.count_nonzero(nominal_controls) == 0
        assert torch.count_nonzero(nominal_disturbances) == 0
        trajectory = states[:, None].expand(states.shape[0], 3, states.shape[-1]).clone()
        return SimpleNamespace(
            states=trajectory,
            suffix_values=torch.tensor([[0.3, 0.2, 0.1]]).expand(states.shape[0], 3),
            score=torch.full((states.shape[0],), 0.3),
        )

    monkeypatch.setattr(experiments, 'optimize_joint_sequences', fake_joint_optimizer)
    model = torch.nn.Linear(3, 1)
    experiment = experiments.DeepReach(model, DatasetStub(), str(tmp_path), use_wandb=False)
    replay = MPCReplayBuffer(state_dim=2, capacity=100)
    config = SimpleNamespace(horizon_steps=2, dt=0.25)

    experiment._refresh_mpc_dataset(
        'cpu', {'attacker': config, 'defender': config}, 2, replay, 'joint')

    assert len(replay) == 6
    assert torch.equal(replay.times, torch.tensor([1.0, 0.75, 0.5, 1.0, 0.75, 0.5]))
    assert torch.equal(replay.values, torch.tensor([0.3, 0.2, 0.1, 0.3, 0.2, 0.1]))
    assert model.training


def test_interception_mpc_states_match_requested_spatial_distributions():
    torch.manual_seed(7)
    dynamics = CrazyflieInterception(
        target_R=0.25, capture_R=0.2, accel_max_a=5.0, accel_max_d=7.0)

    states = experiments.sample_mpc_initial_states(
        dynamics, 5000, distribution='interception',
        defender_position_std=0.5, attacker_boundary_std=0.2)

    assert torch.count_nonzero(states[:, [1, 3]]) > 0
    assert torch.all(torch.abs(states[:, [1, 3]]) <= 3.0)
    assert torch.count_nonzero(states[:, [5, 7]]) == 0
    assert torch.all(torch.abs(states[:, [0, 2, 4, 6]]) <= 2.0)

    defender_radius = torch.linalg.vector_norm(states[:, [4, 6]], dim=-1)
    assert torch.all(defender_radius > dynamics.target_R + dynamics.capture_R)
    assert torch.all(torch.abs(states[:, [4, 6]].mean(dim=0)) < 0.03)

    attacker_edge_distance = 2.0 - torch.amax(torch.abs(states[:, [0, 2]]), dim=-1)
    assert torch.mean((attacker_edge_distance <= 0.4).float()) > 0.94


def test_defender_exclusion_radius_is_configurable():
    dynamics = CrazyflieInterception(
        target_R=0.25, capture_R=0.2, accel_max_a=5.0, accel_max_d=7.0, defender_exclusion_R=0.15)
    state = torch.zeros(1, 8)
    state[0, 0] = 1.5   # attacker far from target and defender
    state[0, 4] = 0.3   # defender inside the old 0.45 ring, outside the new 0.15 ring
    assert torch.allclose(dynamics.reach_fn(state), torch.tensor([0.15]))
    assert dynamics.avoid_fn(state).item() > 0

    torch.manual_seed(3)
    states = experiments.sample_mpc_initial_states(dynamics, 2000, distribution='interception')
    defender_radius = torch.linalg.vector_norm(states[:, [4, 6]], dim=-1)
    assert torch.all(defender_radius > 0.15)
    assert torch.any(defender_radius < 0.45)


def test_inward_attacker_velocity_points_at_target():
    torch.manual_seed(5)
    dynamics = CrazyflieInterception(
        target_R=0.25, capture_R=0.2, accel_max_a=5.0, accel_max_d=7.0)
    states = experiments.sample_mpc_initial_states(
        dynamics, 5000, distribution='interception',
        attacker_velocity='inward', attacker_velocity_spread_deg=60.0)

    position = states[:, [0, 2]]
    velocity = states[:, [1, 3]]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    cosine = -(position * velocity).sum(-1) / (torch.linalg.vector_norm(position, dim=-1) * speed)
    assert torch.all(cosine[speed > 1e-3] >= 0.5 - 1e-4)   # within 60 degrees of the target direction
    assert torch.all(torch.abs(velocity) <= 3.0)


def test_tmax_time_distribution_requires_finished_curriculum():
    from utils.mpc_data import sample_mpc_initial_times
    dataset = SimpleNamespace(tMax=1.0, _current_t_max=lambda: 1.0)
    assert torch.equal(sample_mpc_initial_times(dataset, 3, 'tmax'), torch.ones(3))
    dataset._current_t_max = lambda: 0.5
    try:
        sample_mpc_initial_times(dataset, 3, 'tmax')
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError before the curriculum reaches tMax')


def test_closed_loop_refresh_passes_options_and_labels_executed_trajectory(monkeypatch, tmp_path):
    class DatasetStub:
        dynamics = SimpleNamespace(state_dim=2, control_dim=1, disturbance_dim=1,
                                   input_to_coord=lambda c: c, reach_fn=None, avoid_fn=None,
                                   optimal_control=None, optimal_disturbance=None)

        def _sample_times(self, count):
            return torch.ones(count, 1)

    calls = {}

    def fake_closed_loop(states, times, nominal_u, nominal_d, responder, dynamics,
                         attacker_config, defender_config, optimized_player, **kwargs):
        calls.update(player=optimized_player, **kwargs)
        assert nominal_u.shape == (2, 2, 1) and nominal_d.shape == (2, 2, 1)
        return SimpleNamespace(
            states=states[:, None].expand(2, 3, 2).clone(),
            suffix_values=torch.tensor([[0.3, 0.2, 0.1]]).expand(2, 3),
            score=torch.full((2,), 0.3))

    monkeypatch.setattr(experiments, 'closed_loop_rollout', fake_closed_loop)
    experiment = experiments.DeepReach(torch.nn.Linear(3, 1), DatasetStub(), str(tmp_path), use_wandb=False)
    replay = MPCReplayBuffer(state_dim=2, capacity=100)
    config = SimpleNamespace(horizon_steps=2, dt=0.25)
    experiment._refresh_mpc_dataset(
        'cpu', {'attacker': config, 'defender': config}, 2, replay, 'defender',
        rollout='closed_loop', replan_every=2)

    assert calls['player'] == 'defender' and calls['replan_every'] == 2
    assert torch.equal(replay.times, torch.tensor([1.0, 0.75, 0.5, 1.0, 0.75, 0.5]))
    assert torch.equal(replay.values, torch.tensor([0.3, 0.2, 0.1, 0.3, 0.2, 0.1]))


def test_crop_to_domain_stores_only_in_domain_labels(monkeypatch, tmp_path):
    dynamics = CrazyflieInterception(0.25, 0.2, 5.0, 7.0, defender_exclusion_R=0.15)

    class DatasetStub:
        def __init__(self):
            self.dynamics = dynamics

        def _sample_times(self, count):
            return torch.ones(count, 1)

    def fake_closed_loop(states, times, nominal_u, nominal_d, responder, dyn, *args, **kwargs):
        path = states[:, None].expand(states.shape[0], 3, 8).clone()
        path[:, 2, 0] = 2.5                                   # last state leaves the position domain
        return SimpleNamespace(states=path, suffix_values=torch.tensor([[0.3, 0.2, 0.1]]).expand(states.shape[0], 3),
                               score=torch.full((states.shape[0],), 0.3), event_steps=None)

    monkeypatch.setattr(experiments, 'closed_loop_rollout', fake_closed_loop)
    experiment = experiments.DeepReach(torch.nn.Linear(9, 1), DatasetStub(), str(tmp_path), use_wandb=False)
    replay = MPCReplayBuffer(state_dim=8, capacity=100)
    config = SimpleNamespace(horizon_steps=2, dt=0.25)
    experiment._refresh_mpc_dataset('cpu', {'attacker': config, 'defender': config}, 2, replay, 'joint',
                                    state_distribution='interception', rollout='closed_loop',
                                    crop_to_domain=True, use_network=False, initial_guess='zero')
    assert len(replay) == 4                                       # 2 scenarios x 2 in-domain states
    assert torch.equal(replay.values, torch.tensor([0.3, 0.2, 0.3, 0.2]))   # labels from the full game
