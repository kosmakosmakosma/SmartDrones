import wandb
import copy
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
from controllers.mpc import closed_loop_rollout, optimize_control_sequence, optimize_disturbance_sequence, optimize_joint_sequences, optimize_maxmin_sequences, optimize_mixed_sequences
from utils.mpc_data import (MPCReplayBuffer, abs_error_by_time, in_domain_mask, mpc_label_times, outcome_metrics,
                            sample_mpc_initial_states, sample_mpc_initial_times)
from utils.dataio import POINT_GROUPS


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
        temporary_path = '%s.tmp.%d' % (path, os.getpid())
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

    @staticmethod
    def _load_training_checkpoint(checkpoints_dir, resume_path):
        try:
            return torch.load(resume_path, map_location='cpu', weights_only=False)
        except Exception as error:
            print('Warning: could not load %s (%s)' % (resume_path, error))

        epoch_paths = []
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith('model_epoch_') and filename.endswith('.pth'):
                try:
                    epoch = int(filename[len('model_epoch_'):-len('.pth')])
                except ValueError:
                    continue
                epoch_paths.append((epoch, os.path.join(checkpoints_dir, filename)))

        for _, path in sorted(epoch_paths, reverse=True):
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                print('Recovered resume state from %s' % os.path.basename(path))
                return checkpoint
            except Exception as error:
                print('Warning: could not load fallback checkpoint %s (%s)' % (path, error))

        raise RuntimeError('Cannot resume: no valid training checkpoint found in %s' % checkpoints_dir)

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
            attacker_boundary_std=0.2, attacker_velocity='uniform',
            attacker_velocity_spread_deg=60.0, attacker_speed_max=None,
            time_distribution='uniform', rollout='open_loop', replan_every=1,
            end_on_event=False, game_solver='alternating', use_network=True, crop_to_domain=False,
            labels_per_refresh=None, fixed_start_time=None, log_new_games=True):
        """Play MPC games, label their states and add them to `replay_buffer` (if given).

        Before the labels are added, the network's predictions on the new games are compared with their
        outcomes (logged as mpc_new/*), since the network has not been trained on them yet.
        Returns the kept labels and the games' starting states with their outcomes (CPU tensors).
        """
        dynamics = self.dataset.dynamics
        if getattr(dynamics, 'box_loses', False):
            # the MPC players may leave the box (their labels are cropped to the domain instead);
            # only the network's game ends at the box
            dynamics = copy.copy(dynamics)
            dynamics.box_loses = False
        required_methods = ('reach_fn', 'avoid_fn', 'optimal_control', 'optimal_disturbance')
        if not all(hasattr(dynamics, name) for name in required_methods):
            raise NotImplementedError(
                'MPC guidance requires a dynamics class implementing reach_fn, avoid_fn, '
                'optimal_control, and optimal_disturbance')

        was_training = self.model.training
        requires_grad_flags = [parameter.requires_grad for parameter in self.model.parameters()]
        self.model.eval()
        self.model.requires_grad_(False)

        if fixed_start_time is not None:
            times = torch.full((num_initial_states,), float(fixed_start_time), device=device)
        else:
            times = sample_mpc_initial_times(self.dataset, num_initial_states, time_distribution).to(device)
        real_states = sample_mpc_initial_states(
            dynamics, num_initial_states, state_distribution,
            defender_position_std, attacker_boundary_std, attacker_velocity,
            attacker_velocity_spread_deg, attacker_speed_max).to(device)

        if initial_guess not in ('network', 'zero'):
            raise ValueError("initial_guess must be 'network' or 'zero'")
        if game_solver not in ('alternating', 'maxmin', 'mixed'):
            raise ValueError("game_solver must be 'alternating', 'maxmin' or 'mixed'")
        if not use_network and optimized_player != 'joint':
            raise ValueError('MPC without the network requires optimized_player=joint')
        responder = (NeuralBangBangController(model=self.model, dynamics=dynamics, device=device)
                     if use_network else None)
        initial_query = (responder.query(real_states, times)
                         if use_network and initial_guess == 'network' else None)

        def initial_sequence(network_actions, horizon_steps, action_dim):
            if network_actions is not None:
                return network_actions[:, None, :].expand(
                    num_initial_states, horizon_steps, action_dim).clone()
            return torch.zeros(num_initial_states, horizon_steps, action_dim, device=device)

        if rollout not in ('open_loop', 'closed_loop'):
            raise ValueError("rollout must be 'open_loop' or 'closed_loop'")
        if rollout == 'closed_loop':
            if optimized_player not in ('attacker', 'defender', 'joint'):
                raise ValueError("optimized_player must be 'attacker', 'defender', or 'joint'")
            attacker_config = mpc_configs['attacker']
            defender_config = mpc_configs['defender']
            result = closed_loop_rollout(
                real_states, times,
                initial_sequence(initial_query.controls if initial_query is not None else None,
                                 attacker_config.horizon_steps, dynamics.control_dim),
                initial_sequence(initial_query.disturbances if initial_query is not None else None,
                                 defender_config.horizon_steps, dynamics.disturbance_dim),
                responder, dynamics, attacker_config, defender_config, optimized_player,
                replan_every=replan_every, generator=generator, use_network_terminal_value=use_network,
                end_on_event=end_on_event, game_solver=game_solver,
            )
            mpc_config = defender_config if optimized_player == 'defender' else attacker_config
        elif optimized_player == 'attacker':
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
            solver = {'maxmin': optimize_maxmin_sequences, 'mixed': optimize_mixed_sequences,
                      'alternating': optimize_joint_sequences}[game_solver]
            result = solver(
                real_states, times, nominal_controls, nominal_disturbances,
                responder, dynamics, attacker_config, defender_config,
                generator=generator, use_network_terminal_value=use_network,
            )
            mpc_config = attacker_config
        else:
            raise ValueError("optimized_player must be 'attacker', 'defender', or 'joint'")

        horizon_plus_one = result.states.shape[1]
        label_times = mpc_label_times(dynamics, result, times, mpc_config.dt)

        keep = torch.ones_like(label_times, dtype=torch.bool)
        event_steps = getattr(result, 'event_steps', None)
        if event_steps is not None:   # states after the game ended are frozen duplicates: drop them
            keep = torch.arange(horizon_plus_one, device=label_times.device)[None] <= event_steps[:, None]
        if crop_to_domain:   # labels were computed over the full game; only in-domain states are stored
            keep = keep & in_domain_mask(dynamics, result.states)
        # states after the time-to-go ran out are frozen copies of the last one: keep only the first
        step_times = torch.arange(horizon_plus_one, device=label_times.device)[None] * mpc_config.dt
        keep = keep & (step_times <= times[:, None] + 1e-6)
        if labels_per_refresh is not None and int(keep.sum()) > labels_per_refresh:
            kept = torch.nonzero(keep.reshape(-1)).squeeze(-1)
            chosen = kept[torch.randperm(kept.numel(), device=kept.device)[:labels_per_refresh]]
            keep = torch.zeros_like(keep.reshape(-1)).index_fill_(0, chosen, True).view_as(keep)
        games = {
            'times': label_times[keep].detach().cpu(),
            'states': result.states[keep].reshape(-1, dynamics.state_dim).detach().cpu(),
            'values': result.suffix_values[keep].detach().cpu(),
            'start_times': label_times[:, 0].detach().cpu(),
            'start_states': result.states[:, 0].detach().cpu(),
            'start_values': result.suffix_values[:, 0].detach().cpu(),
        }

        new_game_metrics = {}
        if log_new_games:   # the network has not seen these games yet
            start_outcome = outcome_metrics(
                self._network_values(games['start_times'], games['start_states'], device), games['start_values'])
            new_game_metrics = {'mpc_new/start_%s' % key: value for key, value in start_outcome.items()}
            label_predictions = self._network_values(games['times'], games['states'], device)
            for name, error in abs_error_by_time(label_predictions, games['values'], games['times']).items():
                new_game_metrics['mpc_new/abs_error_%s' % name] = error

        if replay_buffer is not None:
            replay_buffer.add(games['times'], games['states'], games['values'])
            print('%s %s MPC dataset refresh: %d initial states, %d labels added, replay buffer size %d%s' % (
                optimized_player.capitalize(), rollout.replace('_', '-'), num_initial_states, int(keep.sum()),
                len(replay_buffer), '' if not new_game_metrics else
                ', network predicts the winner of new games with balanced accuracy %.2f'
                % new_game_metrics['mpc_new/start_balanced_accuracy']))
            if self.use_wandb:
                wandb.log({
                    'mpc_replay_buffer_size': len(replay_buffer),
                    'mpc_%s_mean_score' % optimized_player: result.score.mean().item(),
                    **new_game_metrics,
                })

        for parameter, required_grad in zip(self.model.parameters(), requires_grad_flags):
            parameter.requires_grad_(required_grad)
        if was_training:
            self.model.train()
        return games

    def _network_values(self, times, states, device, batch_size=20000):
        """V(t, x) of the current network in real units (no gradients), CPU tensor."""
        was_training = self.model.training
        self.model.eval()
        dynamics = self.dataset.dynamics
        values = []
        for start in range(0, states.shape[0], batch_size):
            coords = torch.cat((times[start:start + batch_size].reshape(-1, 1).float(),
                                states[start:start + batch_size].float()), dim=1).to(device)
            if dynamics.input_dim > dynamics.state_dim + 1:
                coords = torch.cat((coords, torch.zeros(coords.shape[0], dynamics.input_dim - dynamics.state_dim - 1,
                                                        device=device)), dim=1)
            with torch.no_grad():
                results = self.model({'coords': dynamics.coord_to_input(coords)})
                values.append(dynamics.io_to_value(results['model_in'], results['model_out'].squeeze(dim=-1)).cpu())
        if was_training:
            self.model.train()
        return torch.cat(values) if values else torch.zeros(0)

    def _prepare_mpc_holdout(self, device, refresh_kwargs, num_games, seed=12345):
        """Fixed held-out MPC games from the full time-to-go, generated once per experiment and cached."""
        path = os.path.join(self.experiment_dir, 'training', 'mpc_holdout.pt')
        if os.path.exists(path):
            holdout = torch.load(path, map_location='cpu', weights_only=False)
            print('Loaded %d held-out MPC games from %s' % (holdout['start_values'].numel(), path))
            return holdout
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):   # same games for every run, training RNG untouched
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            holdout = self._refresh_mpc_dataset(
                device, num_initial_states=num_games, replay_buffer=None, generator=generator,
                fixed_start_time=self.dataset.tMax, log_new_games=False,
                **dict(refresh_kwargs, labels_per_refresh=None))
        self._atomic_torch_save(holdout, path)
        print('Generated %d held-out MPC games (%d labelled states) -> %s' % (
            num_games, holdout['values'].numel(), path))
        return holdout

    def _evaluate_mpc_holdout(self, holdout, device, epoch):
        """Network vs held-out game outcomes, on states whose time-to-go the curriculum has reached."""
        max_time = self.dataset._current_t_max() + 1e-6
        metrics = {'step': epoch}
        start_mask = holdout['start_times'] <= max_time
        if start_mask.any():
            outcome = outcome_metrics(self._network_values(
                holdout['start_times'][start_mask], holdout['start_states'][start_mask], device),
                holdout['start_values'][start_mask])
            metrics.update({'holdout/start_%s' % key: value for key, value in outcome.items()})
        state_mask = holdout['times'] <= max_time
        if state_mask.any():
            predicted = self._network_values(holdout['times'][state_mask], holdout['states'][state_mask], device)
            outcome = outcome_metrics(predicted, holdout['values'][state_mask])
            metrics.update({'holdout/states_%s' % key: value for key, value in outcome.items()})
            for name, error in abs_error_by_time(predicted, holdout['values'][state_mask],
                                                 holdout['times'][state_mask]).items():
                metrics['holdout/abs_error_%s' % name] = error
        if 'holdout/start_balanced_accuracy' in metrics:
            print('Held-out MPC games (epoch %d): winner balanced accuracy %.3f, mean |V - outcome| %.4f' % (
                epoch, metrics['holdout/start_balanced_accuracy'], metrics['holdout/start_mean_abs_error']))
        if self.use_wandb:
            wandb.log(metrics)
        return metrics

    @staticmethod
    def _batch_diagnostics(gt, residual_points, values, model_coords):
        """PDE residual per point group and MPC label errors (overall and per time-to-go) for one batch."""
        metrics = {}
        groups = gt.get('point_group')
        if residual_points is not None and groups is not None:
            for index, name in enumerate(POINT_GROUPS):
                mask = groups == index
                if mask.any():
                    metrics['pde_loss_%s' % name] = residual_points[mask].mean().item()
        mpc_mask = gt.get('mpc_mask')
        if mpc_mask is not None and mpc_mask.any():
            predicted, labels = values[mpc_mask].detach(), gt['mpc_targets'][mpc_mask]
            outcome = outcome_metrics(predicted, labels)
            metrics['mpc_train/mean_error'] = outcome['mean_error']
            metrics['mpc_train/balanced_accuracy'] = outcome['balanced_accuracy']
            for name, error in abs_error_by_time(predicted, labels, model_coords[..., 0][mpc_mask]).items():
                metrics['mpc_train/abs_error_%s' % name] = error
        return metrics

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
        val_slice = getattr(self, 'val_slice', None) or {'kind': 'zero_velocity'}
        interception_slice = val_slice['kind'] == 'interception' and self.dataset.dynamics.state_dim == 8
        attacker_speed = 0.0
        if interception_slice:
            # defender at rest at (px_d, 0) near the target; attacker flying straight at the target
            slice_min, slice_max = -val_slice['defender_range'], val_slice['defender_range']
            attacker_speed = val_slice['attacker_speed']
            vel_max_a = getattr(self.dataset.dynamics, 'vel_max_a', None)
            if vel_max_a is not None:
                attacker_speed = min(attacker_speed, vel_max_a)

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
                if interception_slice:
                    position = xys
                    direction = -position / torch.linalg.vector_norm(position, dim=-1, keepdim=True).clamp_min(1e-6)
                    coords[:, 2] = attacker_speed * direction[:, 0]     # vx_a
                    coords[:, 4] = attacker_speed * direction[:, 1]     # vy_a

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

        if interception_slice:
            slice_description = ('attacker flying at the target at %.1f m/s from each position, '
                                 'defender at rest at (px_d, 0)' % attacker_speed)
        else:
            slice_description = 'Fixed: ' + ', '.join(
                '%s=%.1f' % (plot_config['state_labels'][dim], value)
                for dim, value in enumerate(plot_config['state_slices'])
                if dim not in [x_idx, y_idx, slice_idx])
        fig.suptitle(
            '%s value function (black: V=0)\n%s' %
            (type(self.dataset.dynamics).__name__, slice_description))
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
        model_states = self.dataset._sample_uniform_states(num_candidates)
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
            mpc_attacker_velocity='uniform', mpc_attacker_velocity_spread_deg=60.0,
            mpc_attacker_speed_max=None, mpc_time_distribution='uniform',
            mpc_rollout='open_loop', mpc_replan_every=1, mpc_end_on_event=False,
            mpc_game_solver='alternating', mpc_use_network=True, mpc_crop_to_domain=False,
            mpc_labels_per_refresh=None,
            mpc_loss_weight=1.0, mpc_loss_type='l2', mpc_seed=None, mpc_initial_guess='network',
            mpc_optimized_player='attacker',
            lr_final=None, lr_decay_start_epoch=None, resume_lr=None,
            mpc_holdout_games=0, mpc_holdout_eval_epochs=5000,
            val_slice='zero_velocity', val_attacker_speed=2.0, val_defender_range=1.0,
            value_ceiling_weight=0.0, monotonic_weight=0.0,
        ):
        self.val_slice = dict(kind=val_slice, attacker_speed=val_attacker_speed, defender_range=val_defender_range)
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
            if getattr(self.dataset, 'mpc_fraction', 0) > 0:
                # MPC replay states become part of every batch (PDE points with value labels)
                self.dataset.mpc_sampler = mpc_replay_buffer.sample_up_to_time

        if mpc_loss_type not in ('l1', 'l2'):
            raise ValueError("mpc_loss_type must be 'l1' or 'l2'")

        def mpc_label_loss(errors):
            # l1: every label pulls with constant strength (like the PDE residual); l2: squared error
            return torch.mean(torch.abs(errors)) if mpc_loss_type == 'l1' else torch.mean(errors ** 2)

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
            checkpoint = self._load_training_checkpoint(checkpoints_dir, resume_checkpoint_path)
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
            if resume_lr is not None:
                lr = resume_lr   # the saved optimizer state carries the old learning rate: override it
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                print('Learning rate set to %g for resumed training' % lr)

        target_epochs = epochs + additional_epochs
        if start_epoch > target_epochs:
            raise RuntimeError(
                'Checkpoint epoch %d is beyond requested target epoch %d' %
                (start_epoch, target_epochs))
        if additional_epochs:
            print('Refining at the full horizon through epoch %d' % target_epochs)

        refresh_kwargs = dict(
            mpc_configs=mpc_config, optimized_player='joint' if mpc_optimized_player == 'both' else mpc_optimized_player,
            initial_guess=mpc_initial_guess, state_distribution=mpc_state_distribution,
            defender_position_std=mpc_defender_position_std, attacker_boundary_std=mpc_attacker_boundary_std,
            attacker_velocity=mpc_attacker_velocity, attacker_velocity_spread_deg=mpc_attacker_velocity_spread_deg,
            attacker_speed_max=mpc_attacker_speed_max, time_distribution=mpc_time_distribution,
            rollout=mpc_rollout, replan_every=mpc_replan_every, end_on_event=mpc_end_on_event,
            game_solver=mpc_game_solver, use_network=mpc_use_network, crop_to_domain=mpc_crop_to_domain,
            labels_per_refresh=mpc_labels_per_refresh)
        mpc_holdout = None
        if use_mpc_guidance and mpc_holdout_games > 0:
            mpc_holdout = self._prepare_mpc_holdout(device, refresh_kwargs, mpc_holdout_games)

        def scheduled_lr(epoch):
            """Constant lr until lr_decay_start_epoch, then exponential decay reaching lr_final at the last epoch."""
            if lr_final is None or lr_decay_start_epoch is None or epoch < lr_decay_start_epoch:
                return lr
            span = max(target_epochs - lr_decay_start_epoch, 1)
            progress = min((epoch - lr_decay_start_epoch) / span, 1.0)
            return lr * (lr_final / lr) ** progress

        with tqdm(total=len(train_dataloader) * target_epochs, initial=len(train_dataloader) * start_epoch) as pbar:
            for epoch in range(start_epoch, target_epochs):
                current_lr = scheduled_lr(epoch)
                if lr_final is not None:
                    for param_group in optim.param_groups:
                        param_group['lr'] = current_lr
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
                            device, num_initial_states=mpc_num_initial_states, replay_buffer=mpc_replay_buffer,
                            generator=mpc_generator, **dict(refresh_kwargs, optimized_player=optimized_player))
                if (mpc_holdout is not None and not self.dataset.pretrain and epoch > start_epoch and
                        not epoch % mpc_holdout_eval_epochs):
                    self._evaluate_mpc_holdout(mpc_holdout, device, epoch)
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
                    residual_points = losses.pop('pde_residual_points', None)   # logging only

                    # exact one-sided rules of the game (not during pretraining, where every point is at t = 0)
                    bound_metrics = {}
                    summary_step = not total_steps % steps_til_summary
                    has_ceiling = hasattr(self.dataset.dynamics, 'value_upper_bound')
                    if not self.dataset.pretrain and has_ceiling and (value_ceiling_weight > 0 or summary_step):
                        times = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 0]
                        excess = torch.relu(values - self.dataset.dynamics.value_upper_bound(times, states.detach()))
                        if value_ceiling_weight > 0:
                            losses['value_ceiling'] = value_ceiling_weight * excess.mean()
                        bound_metrics['bounds/ceiling_excess'] = excess.mean().item()
                        bound_metrics['bounds/ceiling_violation_share'] = (excess > 1e-3).float().mean().item()
                        # the same per point group (random / capture / mpc ...) to locate the violations
                        groups = gt.get('point_group')
                        if groups is not None:
                            for index, name in enumerate(POINT_GROUPS):
                                mask = groups == index
                                if mask.any():
                                    bound_metrics['bounds/ceiling_violation_share_%s' % name] = (
                                        excess[mask] > 1e-3).float().mean().item()
                                    bound_metrics['bounds/ceiling_excess_%s' % name] = excess[mask].mean().item()
                    if not self.dataset.pretrain and (monotonic_weight > 0 or summary_step):
                        increase = torch.relu(dvs[..., 0])
                        if monotonic_weight > 0:
                            losses['monotonic'] = monotonic_weight * increase.mean()
                        bound_metrics['bounds/dvdt_positive_mean'] = increase.mean().item()
                        bound_metrics['bounds/dvdt_positive_share'] = (increase > 1e-2).float().mean().item()

                    mpc_mask = gt.get('mpc_mask')
                    if use_mpc_guidance and mpc_mask is not None and bool(mpc_mask.any()):
                        # MPC states are part of the batch (PDE residual above) and also carry value labels
                        losses['mpc_data'] = mpc_loss_weight * mpc_label_loss(
                            values[mpc_mask] - gt['mpc_targets'][mpc_mask])
                    elif use_mpc_guidance and getattr(self.dataset, 'mpc_fraction', 0) == 0 and len(mpc_replay_buffer) > 0:
                        mpc_times, mpc_states, mpc_targets = mpc_replay_buffer.sample(mpc_batch_size, device)
                        mpc_coords = torch.cat((mpc_times.unsqueeze(-1), mpc_states), dim=-1)
                        if self.dataset.dynamics.input_dim > self.dataset.dynamics.state_dim + 1:
                            mpc_coords = torch.cat((
                                mpc_coords,
                                torch.zeros(mpc_coords.shape[0], self.dataset.dynamics.input_dim - self.dataset.dynamics.state_dim - 1, device=device)), dim=1)
                        mpc_model_input = self.dataset.dynamics.coord_to_input(mpc_coords)
                        mpc_results = self.model({'coords': mpc_model_input})
                        mpc_preds = self.dataset.dynamics.io_to_value(mpc_results['model_in'], mpc_results['model_out'].squeeze(dim=-1))
                        losses['mpc_data'] = mpc_loss_weight * mpc_label_loss(mpc_preds - mpc_targets)
                    
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
                                'lr': optim.param_groups[0]['lr'],
                            }
                            if 'mpc_data' in losses:
                                wandb_metrics['mpc_data_loss'] = losses['mpc_data']
                                wandb_metrics['mpc_loss_weight'] = mpc_loss_weight
                            wandb_metrics.update(self._batch_diagnostics(
                                gt, residual_points, values, model_input['model_coords']))
                            wandb_metrics.update(bound_metrics)
                            for name in ('value_ceiling', 'monotonic'):
                                if name in losses:
                                    wandb_metrics['%s_loss' % name] = losses[name].item()
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

        if mpc_holdout is not None:
            self._evaluate_mpc_holdout(mpc_holdout, device, target_epochs)
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