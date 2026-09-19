from dataclasses import dataclass

import torch


@dataclass
class BangBangQuery:
    values: torch.Tensor
    gradients: torch.Tensor
    controls: torch.Tensor
    disturbances: torch.Tensor


class NeuralBangBangController:
    """Batched DeepReach value and optimal-policy queries in real units."""

    def __init__(self, model, dynamics, device):
        self.model = model
        self.dynamics = dynamics
        self.device = torch.device(device)

    def query(self, states, times):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[-1] != self.dynamics.state_dim:
            raise ValueError(
                f"states must have shape [batch, {self.dynamics.state_dim}]"
            )

        times = torch.as_tensor(times, dtype=states.dtype, device=self.device)
        if times.ndim == 0:
            times = times.expand(states.shape[0])
        else:
            times = times.reshape(-1)
            if times.numel() == 1:
                times = times.expand(states.shape[0])
        if times.shape[0] != states.shape[0]:
            raise ValueError("times must be scalar or contain one value per state")

        coordinates = torch.cat((times.unsqueeze(-1), states), dim=-1)
        model_input = self.dynamics.coord_to_input(coordinates)

        with torch.enable_grad():
            model_results = self.model({"coords": model_input})
            model_output = model_results["model_out"].squeeze(dim=-1)
            model_coordinates = model_results["model_in"]
            values = self.dynamics.io_to_value(model_coordinates, model_output)
            derivatives = self.dynamics.io_to_dv(model_coordinates, model_output)

        gradients = derivatives[..., 1:]
        controls = self.dynamics.optimal_control(states, gradients)
        disturbances = self.dynamics.optimal_disturbance(states, gradients)
        return BangBangQuery(
            values=values.detach(),
            gradients=gradients.detach(),
            controls=controls.detach(),
            disturbances=disturbances.detach(),
        )

    def __call__(self, states, times):
        return self.query(states, times)