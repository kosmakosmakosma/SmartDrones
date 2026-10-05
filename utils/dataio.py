import torch
from torch.utils.data import Dataset

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples,
                 learned_boundary_fraction=0.5, geometric_boundary_fraction=0.2, learned_boundary_update_epochs=1000,
                 learned_boundary_candidate_samples=100000, learned_boundary_keep_samples=10000, learned_boundary_buffer_size=100000,
                 capture_fraction=0.0, mpc_fraction=0.0, pretrain_geometric_fraction=0.0):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples
        self.learned_boundary_fraction = learned_boundary_fraction
        self.geometric_boundary_fraction = geometric_boundary_fraction
        self.learned_boundary_update_epochs = learned_boundary_update_epochs
        self.learned_boundary_candidate_samples = learned_boundary_candidate_samples
        self.learned_boundary_keep_samples = learned_boundary_keep_samples
        self.learned_boundary_buffer_size = learned_boundary_buffer_size
        self.learned_boundary_coords = None
        # capture_fraction: share of each post-pretraining batch placed at the capture distance
        # mpc_fraction: share taken from MPC replay states (PDE points that also carry value labels);
        #   requires mpc_sampler(count, max_time) -> (times, states, values) in real units, or None
        # pretrain_geometric_fraction: share of each pretraining batch on the target/exclusion circles
        self.capture_fraction = capture_fraction
        self.mpc_fraction = mpc_fraction
        self.pretrain_geometric_fraction = pretrain_geometric_fraction
        self.mpc_sampler = None

    def add_learned_boundary_samples(self, model_coords):
        model_coords = model_coords.detach().cpu()
        if self.learned_boundary_coords is None:
            self.learned_boundary_coords = model_coords[-self.learned_boundary_buffer_size:]
        else:
            self.learned_boundary_coords = torch.cat((self.learned_boundary_coords, model_coords), dim=0)[-self.learned_boundary_buffer_size:]

    def state_dict(self):
        return {
            'pretrain': self.pretrain,
            'pretrain_counter': self.pretrain_counter,
            'counter': self.counter,
            'learned_boundary_coords': self.learned_boundary_coords,
        }

    def load_state_dict(self, state):
        self.pretrain = state.get('pretrain', self.pretrain)
        self.pretrain_counter = state.get('pretrain_counter', self.pretrain_counter)
        self.counter = state.get('counter', self.counter)
        self.learned_boundary_coords = state.get('learned_boundary_coords', self.learned_boundary_coords)

    def restore_progress_from_epoch(self, completed_epochs):
        """Infer curriculum state for checkpoints created before dataset state was saved."""
        if completed_epochs < 0:
            raise ValueError('completed_epochs must be non-negative')
        curriculum_epochs = completed_epochs
        if self.pretrain:
            self.pretrain_counter = min(completed_epochs, self.pretrain_iters)
            self.pretrain = completed_epochs < self.pretrain_iters
            curriculum_epochs = max(completed_epochs - self.pretrain_iters, 0)
        self.counter = min(self.counter + curriculum_epochs, self.counter_end)

    def _current_t_max(self):
        if self.pretrain:
            return self.tMin
        return self.tMin + (self.tMax - self.tMin) * min(self.counter / self.counter_end, 1.0)

    def _sample_times(self, num_samples):
        if self.pretrain:
            return torch.full((num_samples, 1), self.tMin)
        return self.tMin + torch.zeros(num_samples, 1).uniform_(0, self._current_t_max() - self.tMin)

    def _sample_uniform_states(self, num_samples):
        if hasattr(self.dynamics, 'sample_model_states'):
            return self.dynamics.sample_model_states(num_samples)   # velocities inside speed disks
        return torch.zeros(num_samples, self.dynamics.state_dim).uniform_(-1, 1)

    def _sample_capture_states(self, num_samples):
        """Defender at the capture distance (+-3 cm) from the attacker in a random direction."""
        model_states = self._sample_uniform_states(num_samples)
        states = self.dynamics.input_to_coord(torch.cat((torch.zeros(num_samples, 1), model_states), dim=1))[:, 1:]
        angles = 2 * torch.pi * torch.rand(num_samples)
        radii = self.dynamics.capture_R + 0.03 * torch.randn(num_samples)
        states[:, 4] = states[:, 0] + radii * torch.cos(angles)
        states[:, 6] = states[:, 2] + radii * torch.sin(angles)
        model_states = self.dynamics.coord_to_input(torch.cat((torch.zeros(num_samples, 1), states), dim=1))[:, 1:]
        return torch.clamp(model_states, -1.0, 1.0)

    def _sample_geometric_boundary_states(self, num_samples):
        model_states = self._sample_uniform_states(num_samples)
        if not all(hasattr(self.dynamics, name) for name in ['target_R', 'capture_R']):
            return model_states

        states = self.dynamics.input_to_coord(torch.cat((torch.zeros(num_samples, 1), model_states), dim=1))[:, 1:]
        num_target = num_samples // 2
        num_exclusion = num_samples - num_target

        if num_target:
            angles = 2 * torch.pi * torch.rand(num_target)
            radii = self.dynamics.target_R + 0.03 * torch.randn(num_target)
            states[:num_target, 0] = radii * torch.cos(angles)
            states[:num_target, 2] = radii * torch.sin(angles)

        if num_exclusion:
            angles = 2 * torch.pi * torch.rand(num_exclusion)
            radii = getattr(self.dynamics, 'defender_exclusion_R', self.dynamics.target_R + self.dynamics.capture_R) + 0.03 * torch.randn(num_exclusion)
            states[num_target:, 4] = radii * torch.cos(angles)
            states[num_target:, 6] = radii * torch.sin(angles)

        model_states = self.dynamics.coord_to_input(torch.cat((torch.zeros(num_samples, 1), states), dim=1))[:, 1:]
        return torch.clamp(model_states, -1.0, 1.0)

    def _sample_learned_boundary_coords(self, num_samples):
        if self.learned_boundary_coords is None or len(self.learned_boundary_coords) == 0:
            return None
        indices = torch.randint(len(self.learned_boundary_coords), (num_samples,))
        coords = self.learned_boundary_coords[indices].clone()
        coords[:, 0:1] = torch.clamp(coords[:, 0:1] + 0.01 * self.tMax * torch.randn(num_samples, 1), self.tMin, self._current_t_max())
        coords[:, 1:1+self.dynamics.state_dim] = torch.clamp(
            coords[:, 1:1+self.dynamics.state_dim] + 0.02 * torch.randn(num_samples, self.dynamics.state_dim), -1.0, 1.0)
        return coords

    def __len__(self):
        return 1

    def _sample_mpc_coords(self, num_samples):
        """MPC replay states with time-to-go <= the current curriculum time, as model coords + labels."""
        if num_samples <= 0 or self.mpc_sampler is None:
            return None, None
        sample = self.mpc_sampler(num_samples, self._current_t_max())
        if sample is None:
            return None, None
        times, states, values = (tensor.detach().cpu().float() for tensor in sample)
        coords = self.dynamics.coord_to_input(torch.cat((times.reshape(-1, 1), states), dim=1))
        return coords[:, :1 + self.dynamics.state_dim], values.reshape(-1)

    def __getitem__(self, idx):
        # uniformly sample domain and include coordinates where source is non-zero 
        if self.pretrain:
            num_geometric = int(self.numpoints * self.pretrain_geometric_fraction)
            num_learned = num_capture = num_mpc = 0
        else:
            num_geometric = int(self.numpoints * self.geometric_boundary_fraction)
            num_learned = int(self.numpoints * self.learned_boundary_fraction)
            num_capture = int(self.numpoints * self.capture_fraction)
            num_mpc = int(self.numpoints * self.mpc_fraction)
        learned_coords = self._sample_learned_boundary_coords(num_learned) if num_learned else None
        if learned_coords is None:
            num_learned = 0
        mpc_coords, mpc_values = self._sample_mpc_coords(num_mpc)
        if mpc_coords is None:   # no MPC data yet: fill with uniform points instead
            num_mpc = 0

        num_uniform = self.numpoints - num_geometric - num_learned - num_capture - num_mpc
        model_states = self._sample_uniform_states(num_uniform)
        if num_geometric:
            model_states = torch.cat((model_states, self._sample_geometric_boundary_states(num_geometric)), dim=0)
        if num_capture:
            model_states = torch.cat((model_states, self._sample_capture_states(num_capture)), dim=0)
        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        times = self._sample_times(model_states.shape[0])
        if not self.pretrain:
            # make sure we always have training samples at the initial time (taken from the uniform points)
            times[:min(self.num_src_samples, num_uniform), 0] = self.tMin
        model_coords = torch.cat((times, model_states), dim=1)
        if learned_coords is not None:
            model_coords = torch.cat((model_coords, learned_coords), dim=0)
        mpc_mask = torch.zeros(model_coords.shape[0], dtype=torch.bool)
        mpc_targets = torch.zeros(model_coords.shape[0])
        if mpc_coords is not None:
            model_coords = torch.cat((model_coords, mpc_coords), dim=0)
            mpc_mask = torch.cat((mpc_mask, torch.ones(mpc_coords.shape[0], dtype=torch.bool)))
            mpc_targets = torch.cat((mpc_targets, mpc_values))
        if self.dynamics.input_dim > self.dynamics.state_dim + 1: # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
            model_coords = torch.cat((model_coords, torch.zeros(model_coords.shape[0], self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)      

        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        
        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        mpc_gt = {'mpc_mask': mpc_mask, 'mpc_targets': mpc_targets}
        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, **mpc_gt}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks, **mpc_gt}
        else:
            raise NotImplementedError