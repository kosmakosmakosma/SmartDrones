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
