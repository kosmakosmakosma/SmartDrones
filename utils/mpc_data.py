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

    def state_dict(self):
        return {"times": self.times, "states": self.states, "values": self.values}

    def load_state_dict(self, state):
        self.times = state["times"]
        self.states = state["states"]
        self.values = state["values"]
