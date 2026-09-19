import torch

from controllers.bang_bang import NeuralBangBangController
from dynamics.dynamics import CrazyflieInterception
from utils import modules


def make_model_and_dynamics():
    dynamics = CrazyflieInterception(target_R=0.25, capture_R=0.2, accel_max_a=5.0, accel_max_d=7.0)
    dynamics.deepreach_model = "exact"
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim, out_features=1, type="sine", mode="mlp",
        final_layer_factor=1., hidden_features=32, num_hidden_layers=2,
    )
    return model, dynamics


def test_query_batched_shapes():
    model, dynamics = make_model_and_dynamics()
    controller = NeuralBangBangController(model=model, dynamics=dynamics, device="cpu")

    states = torch.zeros(4, dynamics.state_dim)
    times = torch.full((4,), 0.5)

    result = controller.query(states, times)

    assert result.values.shape == (4,)
    assert result.gradients.shape == (4, dynamics.state_dim)
    assert result.controls.shape == (4, dynamics.control_dim)
    assert result.disturbances.shape == (4, dynamics.disturbance_dim)


def test_query_single_state_and_scalar_time():
    model, dynamics = make_model_and_dynamics()
    controller = NeuralBangBangController(model=model, dynamics=dynamics, device="cpu")

    state = torch.zeros(dynamics.state_dim)
    result = controller.query(state, 0.3)

    assert result.controls.shape == (1, dynamics.control_dim)


def test_query_controls_within_bounds():
    model, dynamics = make_model_and_dynamics()
    controller = NeuralBangBangController(model=model, dynamics=dynamics, device="cpu")

    states = torch.randn(8, dynamics.state_dim)
    times = torch.full((8,), 0.4)
    result = controller.query(states, times)

    assert torch.all(result.controls.abs() <= dynamics.accel_max_a + 1e-5)
    assert torch.all(result.disturbances.abs() <= dynamics.accel_max_d + 1e-5)


def test_query_mismatched_time_batch_raises():
    model, dynamics = make_model_and_dynamics()
    controller = NeuralBangBangController(model=model, dynamics=dynamics, device="cpu")

    states = torch.zeros(3, dynamics.state_dim)
    times = torch.zeros(2)
    try:
        controller.query(states, times)
    except ValueError:
        return
    assert False, "expected ValueError for mismatched batch sizes"
