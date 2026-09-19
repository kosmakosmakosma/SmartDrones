import wandb
import torch
import os
import shutil
import time
import math
import pickle
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
from controllers.bang_bang import NeuralBangBangController
from controllers.mpc import optimize_control_sequence, optimize_disturbance_sequence, optimize_joint_sequences


def sample_mpc_initial_states(
        dynamics, num_samples, distribution='uniform',
        defender_position_std=0.5, attacker_boundary_std=0.2):
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

    exclusion_radius = dynamics.target_R + dynamics.capture_R
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
    return states

class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    @staticmethod
    def _atomic_torch_save(value, path):
        temporary_path = path + '.tmp'
        torch.save(value, temporary_path)
        for attempt in range(5):
            try:
                os.replace(temporary_path, path)
                return True
            except PermissionError:
                if attempt == 4:
                    print('Warning: could not replace checkpoint %s; keeping the previous checkpoint.' % path)
                    try:
                        os.remove(temporary_path)
                    except OSError:
                        pass
                    return False
                time.sleep(0.5)

    def _training_checkpoint(self, epoch, total_steps, optimizer, train_losses, last_CSL_epoch, new_weight, mpc_replay_buffer=None):
        return {
            'epoch': epoch,
            'total_steps': total_steps,
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'last_CSL_epoch': last_CSL_epoch,
            'new_weight': new_weight,
            'dataset': self.dataset.state_dict(),
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'mpc_replay_buffer': mpc_replay_buffer.state_dict() if mpc_replay_buffer is not None else None,
        }

    def _refresh_mpc_dataset(
            self, device, mpc_configs, num_initial_states, replay_buffer,
            optimized_player, generator=None, initial_guess='zero',
            state_distribution='uniform', defender_position_std=0.5,
            attacker_boundary_std=0.2):
        """Generate BRAT MPC labels against the current policy and add bootstrapped trajectory suffixes to `replay_buffer`."""
        dynamics = self.dataset.dynamics
        required_methods = ('reach_fn', 'avoid_fn', 'optimal_control', 'optimal_disturbance')
        if not all(hasattr(dynamics, name) for name in required_methods):
            raise NotImplementedError(
                'MPC guidance requires a dynamics class implementing reach_fn, avoid_fn, '
                'optimal_control, and optimal_disturbance')

        was_training = self.model.training
        requires_grad_flags = [parameter.requires_grad for parameter in self.model.parameters()]
        self.model.eval()
        self.model.requires_grad_(False)

        times = self.dataset._sample_times(num_initial_states).squeeze(-1).to(device)
        real_states = sample_mpc_initial_states(
            dynamics, num_initial_states, state_distribution,
            defender_position_std, attacker_boundary_std).to(device)

        responder = NeuralBangBangController(model=self.model, dynamics=dynamics, device=device)
        if initial_guess not in ('network', 'zero'):
            raise ValueError("initial_guess must be 'network' or 'zero'")
        initial_query = responder.query(real_states, times) if initial_guess == 'network' else None

        def initial_sequence(network_actions, horizon_steps, action_dim):
            if network_actions is not None:
                return network_actions[:, None, :].expand(
                    num_initial_states, horizon_steps, action_dim).clone()
            return torch.zeros(num_initial_states, horizon_steps, action_dim, device=device)

        if optimized_player == 'attacker':
            mpc_config = mpc_configs['attacker']
            nominal_sequence = initial_sequence(
                initial_query.controls if initial_query is not None else None,
                mpc_config.horizon_steps, dynamics.control_dim)
            result = optimize_control_sequence(
                real_states, times, nominal_sequence, responder, dynamics, mpc_config,
                generator=generator, use_network_terminal_value=True,
            )
        elif optimized_player == 'defender':
            mpc_config = mpc_configs['defender']
            nominal_sequence = initial_sequence(
                initial_query.disturbances if initial_query is not None else None,
                mpc_config.horizon_steps, dynamics.disturbance_dim)
            result = optimize_disturbance_sequence(
                real_states, times, nominal_sequence, responder, dynamics, mpc_config,
                generator=generator, use_network_terminal_value=True,
            )
        elif optimized_player == 'joint':
            attacker_config = mpc_configs['attacker']
            defender_config = mpc_configs['defender']
            nominal_controls = initial_sequence(
                initial_query.controls if initial_query is not None else None,
                attacker_config.horizon_steps, dynamics.control_dim)
            nominal_disturbances = initial_sequence(
                initial_query.disturbances if initial_query is not None else None,
                defender_config.horizon_steps, dynamics.disturbance_dim)
            result = optimize_joint_sequences(
                real_states, times, nominal_controls, nominal_disturbances,
                responder, dynamics, attacker_config, defender_config,
                generator=generator, use_network_terminal_value=True,
            )
            mpc_config = attacker_config
        else:
            raise ValueError("optimized_player must be 'attacker', 'defender', or 'joint'")

        horizon_plus_one = result.states.shape[1]
        label_times = torch.stack(
            [torch.clamp(times - step * mpc_config.dt, min=0.0) for step in range(horizon_plus_one)], dim=1)

        replay_buffer.add(
            label_times.reshape(-1).detach().cpu(),
            result.states.reshape(-1, dynamics.state_dim).detach().cpu(),
            result.suffix_values.reshape(-1).detach().cpu(),
        )

        print('%s MPC dataset refresh: %d initial states, %d labels added, replay buffer size %d' % (
            optimized_player.capitalize(), num_initial_states, label_times.numel(), len(replay_buffer)))
        if self.use_wandb:
            wandb.log({
                'mpc_replay_buffer_size': len(replay_buffer),
                'mpc_%s_mean_score' % optimized_player: result.score.mean().item(),
            })

        for parameter, required_grad in zip(self.model.parameters(), requires_grad_flags):
            parameter.requires_grad_(required_grad)
        if was_training:
            self.model.train()

    def validate(self, device, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()
        x_idx = plot_config['x_axis_idx']
        y_idx = plot_config['y_axis_idx']
        slice_idx = plot_config['z_axis_idx']

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[x_idx]
        y_min, y_max = state_test_range[y_idx]
        slice_min, slice_max = state_test_range[slice_idx]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        slice_values = torch.linspace(slice_min, slice_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)

        panel_values = []
        for i in range(len(times)):
            panel_row = []
            for j in range(len(slice_values)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + x_idx] = xys[:, 0]
                coords[:, 1 + y_idx] = xys[:, 1]
                coords[:, 1 + slice_idx] = slice_values[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                panel_row.append(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T)
            panel_values.append(panel_row)

        value_limit = max(np.percentile(np.abs(np.asarray(panel_values)), 99), 1e-8)
        value_norm = matplotlib.colors.TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit)
        fig, axes = plt.subplots(
            len(times), len(slice_values),
            figsize=(3.2*len(slice_values), 3.0*len(times)),
            sharex=True, sharey=True, squeeze=False, constrained_layout=True)

        image = None
        for i, time_value in enumerate(times):
            for j, slice_value in enumerate(slice_values):
                ax = axes[i, j]
                values = panel_values[i][j]
                image = ax.imshow(
                    values, cmap='coolwarm_r', norm=value_norm, origin='lower',
                    extent=(x_min, x_max, y_min, y_max), aspect='equal')
                if values.min() <= 0 <= values.max():
                    ax.contour(xs.numpy(), ys.numpy(), values, levels=[0], colors='black', linewidths=1.2)
                if i == 0:
                    ax.set_title('%s = %.2f' % (plot_config['state_labels'][slice_idx], slice_value))
                if j == 0:
                    ax.set_ylabel('t = %.2f\n%s' % (time_value, plot_config['state_labels'][y_idx]))
                if i == len(times) - 1:
                    ax.set_xlabel(plot_config['state_labels'][x_idx])

        fixed_states = [
            '%s=%.1f' % (plot_config['state_labels'][dim], value)
            for dim, value in enumerate(plot_config['state_slices'])
            if dim not in [x_idx, y_idx, slice_idx]
        ]
        fig.suptitle(
            '%s value function (black: V=0)\nFixed: %s' %
            (type(self.dataset.dynamics).__name__, ', '.join(fixed_states)))
        fig.colorbar(image, ax=axes, shrink=0.9, label='V(t, x): red ≤ 0, blue > 0')
        fig.savefig(save_path)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def _update_learned_boundary_samples(self, device):
        if self.dataset.learned_boundary_candidate_samples <= 0 or self.dataset.learned_boundary_keep_samples <= 0:
            return

        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        num_candidates = self.dataset.learned_boundary_candidate_samples
        keep_count = min(self.dataset.learned_boundary_keep_samples, num_candidates)
        model_states = torch.zeros(num_candidates, self.dataset.dynamics.state_dim).uniform_(-1, 1)
        times = self.dataset._sample_times(num_candidates)
        model_coords = torch.cat((times, model_states), dim=1)
        if self.dataset.dynamics.input_dim > self.dataset.dynamics.state_dim + 1:
            model_coords = torch.cat((
                model_coords,
                torch.zeros(num_candidates, self.dataset.dynamics.input_dim - self.dataset.dynamics.state_dim - 1)), dim=1)

        with torch.no_grad():
            model_results = self.model({'coords': model_coords.to(device)})
            values = self.dataset.dynamics.io_to_value(
                model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
            boundary_indices = torch.topk(torch.abs(values), keep_count, largest=False).indices.detach().cpu()

        self.dataset.add_learned_boundary_samples(model_coords[boundary_indices])
        print('Updated learned-boundary replay buffer with %d samples (%d total)' % (
            keep_count, len(self.dataset.learned_boundary_coords)))

        if self.use_wandb:
            wandb.log({'learned_boundary_buffer_size': len(self.dataset.learned_boundary_coords)})

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def train(
            self, device, batch_size, epochs, lr, 
            steps_til_summary, epochs_til_checkpoint, 
            loss_fn, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            use_CSL, CSL_lr, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size,
            resume=False, autosave_epochs=10, additional_epochs=0,
            use_mpc_guidance=False, mpc_config=None, mpc_replay_buffer=None,
            mpc_num_initial_states=256, mpc_start_epoch=0,
            mpc_refresh_epochs=1000, mpc_batch_size=1000,
            mpc_state_distribution='interception', mpc_defender_position_std=0.5,
            mpc_attacker_boundary_std=0.2, mpc_reset_replay=False,
            mpc_loss_weight=1.0, mpc_seed=None, mpc_initial_guess='network',
            mpc_optimized_player='attacker',
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        mpc_generator = None
        if use_mpc_guidance:
            if mpc_config is None or mpc_replay_buffer is None:
                raise ValueError('use_mpc_guidance requires both mpc_config and mpc_replay_buffer')
            if mpc_optimized_player not in ('attacker', 'defender', 'both', 'joint'):
                raise ValueError("mpc_optimized_player must be 'attacker', 'defender', 'both', or 'joint'")
            if mpc_initial_guess not in ('network', 'zero'):
                raise ValueError("mpc_initial_guess must be 'network' or 'zero'")
            if mpc_start_epoch < 0:
                raise ValueError('mpc_start_epoch must be non-negative')
            if mpc_seed is not None:
                mpc_generator = torch.Generator(device=device)
                mpc_generator.manual_seed(mpc_seed)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        # copy settings from Raissi et al. (2019) and here 
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        if autosave_epochs < 1:
            raise ValueError('autosave_epochs must be at least 1')

        start_epoch = 0
        total_steps = 0
        new_weight = 1
        train_losses = []
        last_CSL_epoch = -1
        resume_checkpoint_path = os.path.join(checkpoints_dir, 'resume_latest.pth')

        if resume:
            if not os.path.exists(resume_checkpoint_path):
                raise RuntimeError('Cannot resume: %s does not exist' % resume_checkpoint_path)
            checkpoint = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            optim.load_state_dict(checkpoint['optimizer'])
            for optimizer_state in optim.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(device)
            start_epoch = checkpoint['epoch']
            total_steps = checkpoint.get('total_steps', start_epoch * len(train_dataloader))
            train_losses = checkpoint.get('train_losses', [])
            last_CSL_epoch = checkpoint.get('last_CSL_epoch', -1)
            new_weight = checkpoint.get('new_weight', 1)
            dataset_state = checkpoint.get('dataset')
            if dataset_state:
                self.dataset.load_state_dict(dataset_state)
            else:
                self.dataset.restore_progress_from_epoch(start_epoch)
                print('Checkpoint has no dataset state; inferred curriculum progress from epoch %d' % start_epoch)
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['numpy_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            if torch.cuda.is_available() and checkpoint.get('cuda_random_state') is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
            if (use_mpc_guidance and not mpc_reset_replay and
                    checkpoint.get('mpc_replay_buffer') is not None):
                mpc_replay_buffer.load_state_dict(checkpoint['mpc_replay_buffer'])
            elif use_mpc_guidance and mpc_reset_replay:
                print('Resetting saved MPC replay buffer for resumed training')
            print('Resuming training from completed epoch %d' % start_epoch)

        target_epochs = epochs + additional_epochs
        if start_epoch > target_epochs:
            raise RuntimeError(
                'Checkpoint epoch %d is beyond requested target epoch %d' %
                (start_epoch, target_epochs))
        if additional_epochs:
            print('Refining at the full horizon through epoch %d' % target_epochs)

        with tqdm(total=len(train_dataloader) * target_epochs, initial=len(train_dataloader) * start_epoch) as pbar:
            for epoch in range(start_epoch, target_epochs):
                if (not self.dataset.pretrain and
                        self.dataset.learned_boundary_fraction > 0 and
                        not epoch % self.dataset.learned_boundary_update_epochs):
                    self._update_learned_boundary_samples(device)
                if (use_mpc_guidance and epoch >= mpc_start_epoch and
                    not self.dataset.pretrain and
                        not epoch % mpc_refresh_epochs):
                    optimized_players = (('attacker', 'defender')
                                         if mpc_optimized_player == 'both'
                                         else (mpc_optimized_player,))
                    for optimized_player in optimized_players:
                        self._refresh_mpc_dataset(
                            device, mpc_config, mpc_num_initial_states,
                            mpc_replay_buffer, optimized_player, mpc_generator,
                            mpc_initial_guess, mpc_state_distribution,
                            mpc_defender_position_std, mpc_attacker_boundary_std)
                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt
                
                # self-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()
                
                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}

                    model_results = self.model({'coords': model_input['model_coords']})

                    states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                    boundary_values = gt['boundary_values']
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']

                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    else:
                        raise NotImplementedError

                    if use_mpc_guidance and len(mpc_replay_buffer) > 0:
                        mpc_times, mpc_states, mpc_targets = mpc_replay_buffer.sample(mpc_batch_size, device)
                        mpc_coords = torch.cat((mpc_times.unsqueeze(-1), mpc_states), dim=-1)
                        if self.dataset.dynamics.input_dim > self.dataset.dynamics.state_dim + 1:
                            mpc_coords = torch.cat((
                                mpc_coords,
                                torch.zeros(mpc_coords.shape[0], self.dataset.dynamics.input_dim - self.dataset.dynamics.state_dim - 1, device=device)), dim=1)
                        mpc_model_input = self.dataset.dynamics.coord_to_input(mpc_coords)
                        mpc_results = self.model({'coords': mpc_model_input})
                        mpc_preds = self.dataset.dynamics.io_to_value(mpc_results['model_in'], mpc_results['model_out'].squeeze(dim=-1))
                        losses['mpc_data'] = mpc_loss_weight * torch.mean((mpc_preds - mpc_targets) ** 2)
                    
                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean() 
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    # Adjust the relative magnitude of the losses if required
                    if self.dataset.dynamics.deepreach_model in ['vanilla', 'diff'] and adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            # # Plot the gradients
                            # import seaborn as sns
                            # import matplotlib.pyplot as plt
                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # fig.savefig('gradient_visualization.png')

                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # grads_dirichlet_normalized = grads_dirichlet * torch.mean(torch.abs(grads_PDE))/torch.mean(torch.abs(grads_dirichlet))
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet_normalized.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # ax.set_xlim([-1000.0, 1000.0])
                            # fig.savefig('gradient_visualization_normalized.png')

                            # Set the new weight according to the paper
                            # num = torch.max(torch.abs(grads_PDE))
                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight*losses['dirichlet']
                        writer.add_scalar('weight_scaling', new_weight, total_steps)

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_name == 'dirichlet':
                            writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                        else:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                        optim.step()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                        if self.use_wandb:
                            wandb_metrics = {
                                'step': epoch,
                                'train_loss': train_loss,
                                'pde_loss': losses['diff_constraint_hom'],
                            }
                            if 'mpc_data' in losses:
                                wandb_metrics['mpc_data_loss'] = losses['mpc_data']
                                wandb_metrics['mpc_loss_weight'] = mpc_loss_weight
                            wandb.log(wandb_metrics)

                    total_steps += 1

                # cost-supervised learning (CSL) phase
                if use_CSL and not self.dataset.pretrain and (epoch-last_CSL_epoch) >= epochs_til_CSL:
                    last_CSL_epoch = epoch
                    
                    # generate CSL datasets
                    self.model.eval()

                    CSL_dataset = scenario_optimization(
                        device=device, model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_samples, 100000), sample_batch_size=min(10*num_CSL_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_coords = torch.cat((CSL_dataset['times'].unsqueeze(-1), CSL_dataset['states']), dim=-1)
                    CSL_costs = CSL_dataset['costs']

                    num_CSL_val_samples = int(0.1*num_CSL_samples)
                    CSL_val_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_val_coords = torch.cat((CSL_val_dataset['times'].unsqueeze(-1), CSL_val_dataset['states']), dim=-1)
                    CSL_val_costs = CSL_val_dataset['costs']

                    CSL_val_tMax_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=self.dataset.tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples # no tStart_generator, since I want all tMax times
                    )
                    CSL_val_tMax_coords = torch.cat((CSL_val_tMax_dataset['times'].unsqueeze(-1), CSL_val_tMax_dataset['states']), dim=-1)
                    CSL_val_tMax_costs = CSL_val_tMax_dataset['costs']
                    
                    self.model.train()

                    # CSL optimizer
                    CSL_optim = torch.optim.Adam(lr=CSL_lr, params=self.model.parameters())

                    # initial CSL val loss
                    CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(device))})
                    CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_errors = CSL_val_preds - CSL_val_costs.to(device)
                    CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    CSL_initial_val_loss = CSL_val_loss
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "CSL_val_loss": CSL_val_loss.item()
                        })

                    # initial self-supervised learning (SSL) val loss
                    # right now, just took code from dataio.py and the SSL training loop above; TODO: refactor all this for cleaner modular code
                    CSL_val_states = CSL_val_coords[..., 1:].to(device)
                    CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                        CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                    CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                    else:
                        NotImplementedError
                    SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "SSL_val_loss": SSL_val_loss.item()
                        })

                    # CSL training loop
                    for CSL_epoch in tqdm(range(max_CSL_epochs)):
                        CSL_idxs = torch.randperm(num_CSL_samples)
                        for CSL_batch in range(math.ceil(num_CSL_samples/CSL_batch_size)):
                            CSL_batch_idxs = CSL_idxs[CSL_batch*CSL_batch_size:(CSL_batch+1)*CSL_batch_size]
                            CSL_batch_coords = CSL_coords[CSL_batch_idxs]

                            CSL_batch_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_batch_coords.to(device))})
                            CSL_batch_preds = self.dataset.dynamics.io_to_value(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_costs = CSL_costs[CSL_batch_idxs].to(device)
                            CSL_batch_errors = CSL_batch_preds - CSL_batch_costs
                            CSL_batch_loss = CSL_loss_weight*torch.mean(torch.pow(CSL_batch_errors, 2))

                            CSL_batch_states = CSL_batch_coords[..., 1:].to(device)
                            CSL_batch_dvs = self.dataset.dynamics.io_to_dv(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_boundary_values = self.dataset.dynamics.boundary_fn(CSL_batch_states)
                            if self.dataset.dynamics.loss_type == 'brat_hjivi':
                                CSL_batch_reach_values = self.dataset.dynamics.reach_fn(CSL_batch_states)
                                CSL_batch_avoid_values = self.dataset.dynamics.avoid_fn(CSL_batch_states)
                            CSL_batch_dirichlet_masks = CSL_batch_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                            if self.dataset.dynamics.loss_type == 'brt_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_dirichlet_masks)
                            elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_reach_values, CSL_batch_avoid_values, CSL_batch_dirichlet_masks)
                            else:
                                NotImplementedError
                            SSL_batch_loss = SSL_batch_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_batch_dirichlet_masks == False))
                            
                            CSL_optim.zero_grad()
                            SSL_batch_loss.backward(retain_graph=True)
                            if (not use_lbfgs) and clip_grad: # no adjust_relative_grads, because I assume even with adjustment, the diff_constraint_hom remains unaffected and the only other loss (dirichlet) is zero
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                            CSL_batch_loss.backward()
                            CSL_optim.step()
                        
                        # evaluate on CSL_val_dataset
                        CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(device))})
                        CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_errors = CSL_val_preds - CSL_val_costs.to(device)
                        CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    
                        CSL_val_states = CSL_val_coords[..., 1:].to(device)
                        CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                        if self.dataset.dynamics.loss_type == 'brat_hjivi':
                            CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                            CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                        CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                        if self.dataset.dynamics.loss_type == 'brt_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                        elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                        else:
                            raise NotImplementedError
                        SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    
                        CSL_val_tMax_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_tMax_coords.to(device))})
                        CSL_val_tMax_preds = self.dataset.dynamics.io_to_value(CSL_val_tMax_results['model_in'], CSL_val_tMax_results['model_out'].squeeze(dim=-1))
                        CSL_val_tMax_errors = CSL_val_tMax_preds - CSL_val_tMax_costs.to(device)
                        CSL_val_tMax_loss = torch.mean(torch.pow(CSL_val_tMax_errors, 2))
                        
                        # log CSL losses, recovered_safe_set_fracs
                        if self.dataset.dynamics.set_mode == 'reach':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(device) < 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds < torch.min(CSL_batch_preds[CSL_batch_costs.to(device) > 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(device) < 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds < torch.min(CSL_val_preds[CSL_val_costs.to(device) > 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(device) < 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds < torch.min(CSL_val_tMax_preds[CSL_val_tMax_costs.to(device) > 0])) / len(CSL_val_tMax_preds)
                        elif self.dataset.dynamics.set_mode == 'avoid':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(device) > 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(CSL_batch_preds[CSL_batch_costs.to(device) < 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(device) > 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds > torch.max(CSL_val_preds[CSL_val_costs.to(device) < 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(device) > 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds > torch.max(CSL_val_tMax_preds[CSL_val_tMax_costs.to(device) < 0])) / len(CSL_val_tMax_preds)
                        else:
                            raise NotImplementedError
                        if self.use_wandb:
                            wandb.log({
                                "step": epoch+(CSL_epoch+1)*int(0.5*epochs_til_CSL/max_CSL_epochs),
                                "CSL_train_batch_loss": CSL_batch_loss.item(),
                                "SSL_train_batch_loss": SSL_batch_loss.item(),
                                "CSL_val_loss": CSL_val_loss.item(),
                                "SSL_val_loss": SSL_val_loss.item(),
                                "CSL_val_tMax_loss": CSL_val_tMax_loss.item(),
                                "CSL_train_batch_theoretically_recoverable_safe_set_frac": CSL_train_batch_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_theoretically_recoverable_safe_set_frac": CSL_val_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_tMax_theoretically_recoverable_safe_set_frac": CSL_val_tMax_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_train_batch_recovered_safe_set_frac": CSL_train_batch_recovered_safe_set_frac.item(),
                                "CSL_val_recovered_safe_set_frac": CSL_val_recovered_safe_set_frac.item(),
                                "CSL_val_tMax_recovered_safe_set_frac": CSL_val_tMax_recovered_safe_set_frac.item(),
                            })

                        if CSL_val_loss < CSL_loss_frac_cutoff*CSL_initial_val_loss:
                            break

                completed_epochs = epoch + 1
                checkpoint = self._training_checkpoint(
                    completed_epochs, total_steps, optim, train_losses, last_CSL_epoch, new_weight, mpc_replay_buffer)
                if (completed_epochs == 1 or
                        not completed_epochs % autosave_epochs or
                        not completed_epochs % epochs_til_checkpoint):
                    self._atomic_torch_save(checkpoint, resume_checkpoint_path)

                if not completed_epochs % epochs_til_checkpoint:
                    self._atomic_torch_save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % completed_epochs))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % completed_epochs),
                        np.array(train_losses))
                    self.validate(
                        device=device, epoch=completed_epochs, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % completed_epochs),
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

        final_checkpoint = self._training_checkpoint(
            target_epochs, total_steps, optim, train_losses, last_CSL_epoch, new_weight, mpc_replay_buffer)
        self._atomic_torch_save(final_checkpoint, resume_checkpoint_path)
        self._atomic_torch_save(self.model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        writer.close()

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, device, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        if os.path.exists(testing_dir):
            overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
            if not (overwrite == 'y'):
                print('Exiting.')
                quit()
            shutil.rmtree(testing_dir)
        os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                raise NotImplementedError

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics
            raise NotImplementedError

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

class DeepReach(Experiment):
    def init_special(self):
        pass