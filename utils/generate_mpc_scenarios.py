"""Generate MPC interception scenarios outside training and save them for utils/view_mpc_trajectories.py.

Loads a trained experiment (or, with --stand_in, uses boundary_fn instead of a network), samples
interception initial states, runs the MPC exactly like the training refresh does, and writes a
checkpoint-style file whose 'mpc_replay_buffer' holds the labelled trajectories.

Example:
  python -m utils.generate_mpc_scenarios --experiment_name crazyflie_2d_no_grav_mpc_test1 --output mpc_scenarios.pth
  python -m utils.view_mpc_trajectories mpc_scenarios.pth --num-initial-states 300 --horizon-steps 50
"""

import argparse
import inspect
import os
import pickle

import torch

from controllers.bang_bang import NeuralBangBangController
from controllers.mpc import MPCConfig, closed_loop_rollout, optimize_joint_sequences, optimize_maxmin_sequences, optimize_mixed_sequences
from dynamics import dynamics as dynamics_module
from utils.mpc_data import in_domain_mask, mpc_domain_constraint, sample_mpc_initial_states


def load_experiment(experiments_dir, experiment_name, checkpoint, device, defender_exclusion_R,
                    vel_max_a=None, vel_max_d=None, accel_max=None):
    from utils import modules
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'rb') as file:
        orig_opt = pickle.load(file)
    dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
    params = {name: getattr(orig_opt, name)
              for name, param in inspect.signature(dynamics_class).parameters.items()
              if name != 'self' and (hasattr(orig_opt, name) or param.default is inspect.Parameter.empty)}
    if defender_exclusion_R is not None:
        params['defender_exclusion_R'] = defender_exclusion_R
    if 'vel_max_a' in inspect.signature(dynamics_class).parameters:
        params.update(vel_max_a=vel_max_a, vel_max_d=vel_max_d)
    if accel_max is not None:
        params.update(accel_max_a=accel_max, accel_max_d=accel_max)
    dynamics = dynamics_class(**params)
    dynamics.deepreach_model = orig_opt.deepreach_model

    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
        final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
    checkpoints_dir = os.path.join(experiment_dir, 'training', 'checkpoints')
    if checkpoint == -1:
        state = torch.load(os.path.join(checkpoints_dir, 'model_final.pth'), map_location=device)
    else:
        state = torch.load(os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % checkpoint),
                           map_location=device)['model']
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return dynamics, NeuralBangBangController(model=model, dynamics=dynamics, device=device), orig_opt.tMax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', default='./runs')
    parser.add_argument('--experiment_name', default=None, help='Trained experiment providing the network')
    parser.add_argument('--checkpoint', type=int, default=-1, help='-1 for model_final.pth, else epoch number')
    parser.add_argument('--no_network', action='store_true',
                        help='Joint MPC without any network: zero initial plans, terminal-set margin as terminal value '
                             '(an experiment is still used for its dynamics parameters and tMax if given)')
    parser.add_argument('--game_solver', default='maxmin', choices=['maxmin', 'mixed', 'alternating'],
                        help="'maxmin': each player keeps its best worst-case plan over all plan pairs; "
                             "'mixed': samples each player's plan from the matrix-game equilibrium mixture; "
                             "'alternating': best response to the opponent's current plan")
    parser.add_argument('--stand_in', action='store_true',
                        help='Use boundary_fn instead of a trained network (no experiment needed)')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='mpc_scenarios.pth')
    parser.add_argument('--defender_exclusion_R', type=float, default=0.15)
    parser.add_argument('--accel_max', type=float, default=5.0,
                        help='Acceleration limit per axis in m/s^2, the same for both players '
                             '(<= 0: keep the experiment values, or 5 / 7 without an experiment)')
    parser.add_argument('--vel_max_d', type=float, default=3.0, help='Defender speed limit in m/s (<= 0: unlimited)')
    parser.add_argument('--vel_max_a_ratio', type=float, default=1.0,
                        help='Attacker speed limit as a fraction of the defender limit')
    parser.add_argument('--tMax', type=float, default=None, help='Time-to-go of every scenario (default: experiment tMax)')
    parser.add_argument('--num_initial_states', type=int, default=300)
    parser.add_argument('--optimized_player', default='joint', choices=['attacker', 'defender', 'joint'])
    parser.add_argument('--rollout', default='closed_loop', choices=['closed_loop', 'open_loop'])
    parser.add_argument('--replan_every', type=int, default=1)
    parser.add_argument('--defender_keep_out', action=argparse.BooleanOptionalAction, default=True,
                        help='Reject defender MPC plans that enter the defender exclusion zone')
    parser.add_argument('--domain_constraint', default='none', choices=['none', 'position', 'state'],
                        help='Reject MPC candidates whose own drone leaves the training domain before the game ends')
    parser.add_argument('--end_on_event', action=argparse.BooleanOptionalAction, default=True,
                        help='Stop each scenario at capture, target hit or exclusion breach (closed loop only)')
    parser.add_argument('--initial_guess', default='network', choices=['network', 'zero'])
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--horizon_steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--control_hold_steps', type=int, default=10)
    parser.add_argument('--noise_fraction', type=float, default=0.25)
    parser.add_argument('--integrator', default='euler', choices=['euler', 'rk4'])
    parser.add_argument('--attacker_velocity', default='inward', choices=['inward', 'uniform'])
    parser.add_argument('--defender_position_std', type=float, default=0.5)
    parser.add_argument('--attacker_boundary_std', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    accel_max = args.accel_max if args.accel_max > 0 else None
    accel_a, accel_d = (accel_max, accel_max) if accel_max is not None else (5.0, 7.0)
    vel_max_d = args.vel_max_d if args.vel_max_d > 0 else None
    vel_max_a = None if vel_max_d is None else args.vel_max_a_ratio * vel_max_d
    if args.no_network and args.experiment_name is None:
        dynamics = dynamics_module.CrazyflieInterception(
            0.25, 0.2, accel_a, accel_d, defender_exclusion_R=args.defender_exclusion_R,
            vel_max_a=vel_max_a, vel_max_d=vel_max_d)
        responder, t_max, device = None, 1.0, 'cpu'
    elif args.stand_in:
        from utils.benchmark_mpc import BoundaryResponder
        dynamics = dynamics_module.CrazyflieInterception(
            0.25, 0.2, accel_a, accel_d, defender_exclusion_R=args.defender_exclusion_R,
            vel_max_a=vel_max_a, vel_max_d=vel_max_d)
        responder, t_max, device = BoundaryResponder(dynamics), 1.0, 'cpu'
    else:
        if args.experiment_name is None:
            parser.error('--experiment_name is required unless --stand_in is given')
        device = args.device
        dynamics, responder, t_max = load_experiment(
            args.experiments_dir, args.experiment_name, args.checkpoint, device, args.defender_exclusion_R,
            vel_max_a, vel_max_d, accel_max)
    if args.no_network:
        responder = None
        if args.optimized_player != 'joint':
            parser.error('--no_network requires --optimized_player joint')
    t_max = args.tMax if args.tMax is not None else t_max
    if args.horizon_steps * args.dt < t_max - 1e-9:
        print('Warning: horizon %.2fs is shorter than time-to-go %.2fs; the network value closes the tail'
              % (args.horizon_steps * args.dt, t_max))

    torch.manual_seed(args.seed)
    states = sample_mpc_initial_states(
        dynamics, args.num_initial_states, 'interception', args.defender_position_std,
        args.attacker_boundary_std, args.attacker_velocity).to(device)
    times = torch.full((args.num_initial_states,), float(t_max), device=device)

    def config(bound, dim, player):
        return MPCConfig(
            dt=args.dt, horizon_steps=args.horizon_steps, num_samples=args.num_samples,
            num_iterations=args.iterations, noise_std=args.noise_fraction * bound,
            control_lower=torch.full((dim,), -bound), control_upper=torch.full((dim,), bound),
            integration_method=args.integrator, control_hold_steps=args.control_hold_steps,
            **mpc_domain_constraint(dynamics, player, args.domain_constraint, args.defender_keep_out))
    attacker_config = config(dynamics.accel_max_a, dynamics.control_dim, 'attacker')
    defender_config = config(dynamics.accel_max_d, dynamics.disturbance_dim, 'defender')

    if args.initial_guess == 'network' and responder is not None:
        query = responder.query(states, times)
        nominal_u = query.controls[:, None].expand(-1, args.horizon_steps, -1).clone()
        nominal_d = query.disturbances[:, None].expand(-1, args.horizon_steps, -1).clone()
    else:
        nominal_u = torch.zeros(args.num_initial_states, args.horizon_steps, dynamics.control_dim, device=device)
        nominal_d = torch.zeros(args.num_initial_states, args.horizon_steps, dynamics.disturbance_dim, device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    if args.rollout == 'closed_loop':
        result = closed_loop_rollout(
            states, times, nominal_u, nominal_d, responder, dynamics, attacker_config, defender_config,
            args.optimized_player, replan_every=args.replan_every, generator=generator,
            use_network_terminal_value=responder is not None, end_on_event=args.end_on_event,
            game_solver=args.game_solver)
    elif args.optimized_player == 'joint':
        solver = {'maxmin': optimize_maxmin_sequences, 'mixed': optimize_mixed_sequences,
                  'alternating': optimize_joint_sequences}[args.game_solver]
        result = solver(
            states, times, nominal_u, nominal_d, responder, dynamics, attacker_config, defender_config,
            generator=generator, use_network_terminal_value=responder is not None)
    else:
        parser.error("open_loop is only supported here for --optimized_player joint")

    steps = result.states.shape[1]
    label_times = torch.stack([torch.clamp(times - k * args.dt, min=0.0) for k in range(steps)], dim=1)
    # Labels cover the full game; training stores only in-domain states (--mpc_crop_to_domain).
    # The file keeps whole trajectories for the viewer plus the mask of states training would keep.
    in_domain = in_domain_mask(dynamics, result.states)
    live = torch.ones_like(in_domain)
    if result.event_steps is not None:
        live = torch.arange(steps, device=in_domain.device)[None] <= result.event_steps[:, None]
    torch.save({
        'epoch': -1 if (args.stand_in or args.no_network) else args.checkpoint,
        'mpc_replay_buffer': {
            'times': label_times.reshape(-1).cpu(),
            'states': result.states.reshape(-1, dynamics.state_dim).cpu(),
            'values': result.suffix_values.reshape(-1).cpu(),
        },
        'in_domain': in_domain.reshape(-1).cpu(),
        'generation_args': vars(args),
        'scenario_geometry': {
            'target_R': dynamics.target_R, 'capture_R': dynamics.capture_R,
            'defender_exclusion_R': dynamics.defender_exclusion_R,
            'position_bound': float(dynamics.state_var[0]), 'velocity_bound': float(dynamics.state_var[1]),
            'vel_max_a': dynamics.vel_max_a, 'vel_max_d': dynamics.vel_max_d,
            'accel_max_a': dynamics.accel_max_a, 'accel_max_d': dynamics.accel_max_d,
        },
    }, args.output)
    attacker_wins = (result.score <= 0).float().mean().item()
    print('Saved %d scenarios x %d states to %s (attacker succeeds in %.0f%%, mean score %.3f)' % (
        args.num_initial_states, steps, os.path.abspath(args.output), 100 * attacker_wins,
        result.score.mean().item()))
    print('Scenarios leaving the domain: %.0f%%; game states inside the domain (kept for training): %.0f%%' % (
        100 * ((~in_domain) & live).any(dim=1).float().mean().item(),
        100 * (in_domain & live).sum().item() / max(live.sum().item(), 1)))
    print('View with: python -m utils.view_mpc_trajectories %s --num-initial-states %d --horizon-steps %d' % (
        args.output, args.num_initial_states, steps - 1))


if __name__ == '__main__':
    main()
