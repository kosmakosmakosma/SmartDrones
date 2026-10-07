import pytest
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
        'cpu', {'attacker': config, 'defender': config}, 2, replay, 'joint', log_new_games=False)

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
        rollout='closed_loop', replan_every=2, log_new_games=False)

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
                                    crop_to_domain=True, use_network=False, initial_guess='zero',
                                    log_new_games=False)
    assert len(replay) == 4                                       # 2 scenarios x 2 in-domain states
    assert torch.equal(replay.values, torch.tensor([0.3, 0.2, 0.3, 0.2]))   # labels from the full game


def test_speed_limited_hamiltonian_matches_brute_force():
    torch.manual_seed(0)
    dyn = CrazyflieInterception(0.2, 0.2, 5.0, 5.0, defender_exclusion_R=0.2, vel_max_a=3.6, vel_max_d=3.0)
    states = torch.randn(64, 8)
    for vx, vy, vmax in ((1, 3, 3.6), (5, 7, 3.0)):      # put every drone exactly at its speed limit
        v = states[:, [vx, vy]]
        v = v / v.norm(dim=-1, keepdim=True) * vmax
        states[:, vx], states[:, vy] = v[:, 0], v[:, 1]
    dvds = torch.randn(64, 8)
    grid = torch.linspace(-5.0, 5.0, 401)
    u = torch.stack(torch.meshgrid(grid, grid, indexing='ij'), -1).reshape(-1, 2)        # acceleration box

    def best(gradient, velocity, minimize):
        direction = velocity / velocity.norm()
        effective = u - torch.clamp(u @ direction, min=0)[:, None] * direction
        values = effective @ gradient
        return values.min() if minimize else values.max()

    expected = (dvds[:, [0, 2, 4, 6]] * states[:, [1, 3, 5, 7]]).sum(-1)
    expected = expected + torch.stack([best(dvds[i, [1, 3]], states[i, [1, 3]], True) for i in range(64)])
    expected = expected + torch.stack([best(dvds[i, [5, 7]], states[i, [5, 7]], False) for i in range(64)])
    # the grid can only approach the exact optimum, so the formula may be slightly better than the grid
    assert torch.allclose(dyn.hamiltonian(states, dvds), expected, atol=0.05)
    # below the limit it is the plain bang-bang Hamiltonian
    slow = states.clone(); slow[:, [1, 3, 5, 7]] *= 0.5
    plain = CrazyflieInterception(0.2, 0.2, 5.0, 5.0, defender_exclusion_R=0.2)
    assert torch.allclose(dyn.hamiltonian(slow, dvds), plain.hamiltonian(slow, dvds))


def test_box_rule_disk_sampling_and_velocity_normalisation():
    dyn = CrazyflieInterception(0.2, 0.2, 5.0, 5.0, defender_exclusion_R=0.2, vel_max_a=3.6, vel_max_d=3.0,
                                box_loses=True)
    assert torch.allclose(dyn.state_var, torch.tensor([2.0, 3.6, 2.0, 3.6, 2.0, 3.0, 2.0, 3.0]))
    state = torch.tensor([[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]])
    out_a = state.clone(); out_a[0, 0] = 2.1
    out_d = state.clone(); out_d[0, 4] = -2.1
    assert dyn.avoid_fn(out_a).item() < 0 and dyn.boundary_fn(out_a).item() > 0      # attacker out: attacker loses
    assert dyn.reach_fn(out_d).item() < 0 and dyn.boundary_fn(out_d).item() < 0      # defender out: attacker wins
    model_states = dyn.sample_model_states(20000)
    real = dyn.input_to_coord(torch.cat((torch.zeros(20000, 1), model_states), 1))[:, 1:]
    assert real[:, [1, 3]].norm(dim=-1).max() <= 3.6 + 1e-5 and real[:, [5, 7]].norm(dim=-1).max() <= 3.0 + 1e-5


def test_batch_composition_capture_mpc_and_pretraining():
    from utils.dataio import ReachabilityDataset
    dyn = CrazyflieInterception(0.2, 0.2, 5.0, 5.0, defender_exclusion_R=0.2, vel_max_a=3.6, vel_max_d=3.0)
    dataset = ReachabilityDataset(dyn, 600, pretrain=True, pretrain_iters=1, tMin=0.0, tMax=2.0, counter_start=1,
                                  counter_end=10, num_src_samples=10, num_target_samples=0,
                                  learned_boundary_fraction=0.0, geometric_boundary_fraction=0.0,
                                  capture_fraction=1 / 6, mpc_fraction=1 / 3, pretrain_geometric_fraction=0.25)
    replay = MPCReplayBuffer(state_dim=8, capacity=1000)
    replay.add(torch.tensor([0.1, 1.9]), torch.zeros(2, 8), torch.tensor([0.5, -0.5]))
    dataset.mpc_sampler = replay.sample_up_to_time
    inputs, gt = dataset[0]                                        # pretraining batch
    assert inputs['model_coords'].shape[0] == 600 and not gt['mpc_mask'].any()
    assert torch.all(inputs['model_coords'][:, 0] == 0)
    inputs, gt = dataset[0]                                        # curriculum batch with t_max = 0.2
    coords = inputs['model_coords']
    # only one label is eligible (time 0.1 <= 0.2): it is used once, the other MPC slots become uniform points
    assert coords.shape[0] == 600 and int(gt['mpc_mask'].sum()) == 1
    assert torch.allclose(gt['mpc_targets'][gt['mpc_mask']], torch.tensor([0.5]))
    replay.add(torch.full((500,), 0.05), torch.zeros(500, 8), torch.full((500,), -0.3))
    inputs, gt = dataset[0]
    coords = inputs['model_coords']
    assert coords.shape[0] == 600 and int(gt['mpc_mask'].sum()) == 200
    assert torch.all(coords[gt['mpc_mask'], 0] <= dataset._current_t_max() + 1e-6)   # only labels inside the curriculum
    real = dyn.input_to_coord(coords)[:, 1:]
    capture = real[300:400]                                       # uniform 300, then capture 100, then MPC 200
    assert torch.all(((capture[:, [0, 2]] - capture[:, [4, 6]]).norm(dim=-1) - 0.2).abs() < 0.2)
    assert torch.all(coords[:10, 0] == 0)                         # t = 0 points come from the uniform block


def test_replay_sampling_has_no_duplicates_and_outcome_metrics():
    from utils.mpc_data import abs_error_by_time, outcome_metrics
    replay = MPCReplayBuffer(state_dim=1, capacity=100)
    replay.add(torch.linspace(0, 1, 50), torch.arange(50.0).reshape(-1, 1), torch.zeros(50))
    times, states, _ = replay.sample_up_to_time(40, 1.0)
    assert len(set(states.reshape(-1).tolist())) == 40
    assert replay.sample_up_to_time(40, 0.1)[0].numel() == 5      # only 5 labels have time <= 0.1
    # 3 attacker wins (label <= 0), 1 defender win; the prediction gets 2 of 3 attacker wins and the defender win
    metrics = outcome_metrics(torch.tensor([-0.1, -0.2, 0.3, 0.4]), torch.tensor([-0.1, -0.1, -0.1, 0.2]))
    assert abs(metrics['attacker_win_accuracy'] - 2 / 3) < 1e-6 and metrics['defender_win_accuracy'] == 1.0
    assert abs(metrics['balanced_accuracy'] - 5 / 6) < 1e-6 and metrics['attacker_win_share'] == 0.75
    bins = abs_error_by_time(torch.tensor([0.1, 0.5]), torch.tensor([0.0, 0.0]), torch.tensor([0.2, 1.7]))
    assert bins == {'t0.0-0.5': pytest.approx(0.1), 't1.5-max': pytest.approx(0.5)}


def test_label_times_are_measured_to_the_event():
    from utils.mpc_data import mpc_label_times
    dyn = CrazyflieInterception(0.2, 0.2, 5.0, 5.0, defender_exclusion_R=0.2)
    states = torch.zeros(2, 6, 8)
    states[..., 0] = 1.5                      # attacker far from the target
    states[..., 4] = -1.5                     # defender far away: no event
    states[0, 3:, 4] = 1.6                    # scenario 0: defender within capture distance from step 3 on
    result = SimpleNamespace(states=states, event_steps=torch.tensor([3, 5]))
    times = mpc_label_times(dyn, result, torch.tensor([1.0, 0.1]), 0.02)
    assert torch.allclose(times[0, :4], torch.tensor([0.06, 0.04, 0.02, 0.0]))    # counted down to the capture
    assert torch.allclose(times[1], torch.clamp(0.1 - 0.02 * torch.arange(6), min=0.0))   # ran out of time



def test_network_values_new_game_metrics_and_holdout(tmp_path):
    from utils import modules
    from utils.dataio import ReachabilityDataset
    from utils.benchmark_mpc import make_config
    torch.manual_seed(0)
    dyn = CrazyflieInterception(0.2, 0.2, 5.0, 5.0, defender_exclusion_R=0.2, vel_max_a=3.6, vel_max_d=3.0)
    dyn.deepreach_model = 'exact'
    dataset = ReachabilityDataset(dyn, 100, pretrain=False, pretrain_iters=0, tMin=0.0, tMax=0.4, counter_start=10,
                                  counter_end=10, num_src_samples=0, num_target_samples=0,
                                  learned_boundary_fraction=0.0, geometric_boundary_fraction=0.0)
    model = modules.SingleBVPNet(in_features=9, out_features=1, type='sine', mode='mlp', final_layer_factor=1.,
                                 hidden_features=16, num_hidden_layers=1)
    experiment = experiments.DeepReach(model, dataset, str(tmp_path), use_wandb=False)
    (tmp_path / 'training').mkdir()
    common = dict(dt=0.02, horizon_steps=20, num_samples=4, num_iterations=1, noise_fraction=0.25, hold_steps=5, chunk=None)
    configs = {'attacker': make_config(5.0, **common), 'defender': make_config(5.0, **common)}
    kwargs = dict(mpc_configs=configs, optimized_player='joint', initial_guess='zero', state_distribution='interception',
                  attacker_velocity='inward', time_distribution='current_max', rollout='closed_loop', replan_every=5,
                  end_on_event=True, game_solver='alternating', use_network=False, crop_to_domain=True)
    replay = MPCReplayBuffer(state_dim=8, capacity=1000)
    games = experiment._refresh_mpc_dataset('cpu', num_initial_states=6, replay_buffer=replay, **kwargs)
    assert len(replay) == games['values'].numel() > 0
    assert torch.allclose(games['start_times'][games['start_times'] > 0].max(), torch.tensor(0.4))   # current_max start
    # the network value helper agrees with the model evaluated directly at t = 0 (exact model: V = boundary)
    values = experiment._network_values(torch.zeros(5), games['states'][:5], 'cpu')
    assert torch.allclose(values, dyn.boundary_fn(games['states'][:5]), atol=1e-5)
    holdout = experiment._prepare_mpc_holdout('cpu', kwargs, 4)
    assert (tmp_path / 'training' / 'mpc_holdout.pt').exists() and holdout['start_values'].numel() == 4
    again = experiment._prepare_mpc_holdout('cpu', kwargs, 4)          # cached: identical games
    assert torch.equal(again['start_states'], holdout['start_states'])
    metrics = experiment._evaluate_mpc_holdout(holdout, 'cpu', epoch=7)
    assert 0.0 <= metrics['holdout/start_balanced_accuracy'] <= 1.0
