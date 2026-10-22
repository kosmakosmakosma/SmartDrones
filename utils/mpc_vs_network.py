"""Same start states, two games each, shown side by side: MPC vs MPC, and MPC attacker vs network defender.

Both games use the library's MPC for the attacker: every --replan_every steps it plans the whole remaining game for
both players with the max-min solver (pursuit plans, the defender's chase plans, the run's box rule, no network),
and executes its attacker plan. In the first game the planner's defender plan is executed too; in the second the
defender is the trained network (gradient or lookahead controller at a fixed time-to-go, with the safety filter),
which the attacker knows nothing about. Games are played to the end and V is the game's reach-avoid value, as for
the library's labels (> 0: defender wins); the first capture, target hit, keep-out breach or box exit is reported and
the paths are drawn up to it.

Figure (one column per scenario): top row MPC vs MPC, middle row MPC vs network, bottom row the network's value
V(time-to-go, x(t)) along both games. Starts come from --start (repeatable) or are sampled like the library's
'engagement' starts.

  .\\env\\Scripts\\python.exe -m utils.mpc_vs_network --experiment_name crazyflie_2d_mpc_3s_v6 --checkpoint 240000 ^
      --setting 48x12 --defender_policy lookahead --num_scenarios 5
"""

import argparse
import os
import pickle
from dataclasses import replace
from types import SimpleNamespace

import torch

from controllers.mpc import integrate_step, optimize_maxmin_sequences, reach_avoid_suffix_values
from controllers.network_defender import NetworkDefender
from utils.compare_mpc import make_configs, parse_setting, setting_name
from utils.mpc_data import sample_mpc_initial_states
from utils.query_lookahead import load

EVENTS = ('target hit', 'defender in keep-out', 'defender left box', 'captured', 'attacker left box')
ATTACKER_WINS = {'target hit', 'defender in keep-out', 'defender left box'}


def first_events(dynamics, states):
    """[B] index into EVENTS of the event happening at states [B, S], or -1."""
    position_a, position_d = states[:, [0, 2]], states[:, [4, 6]]
    happened = [
        torch.linalg.vector_norm(position_a, dim=-1) <= dynamics.target_R,
        torch.linalg.vector_norm(position_d, dim=-1) <= dynamics.defender_exclusion_R,
        getattr(dynamics, 'box_loses', False) & (position_d.abs().amax(-1) > dynamics.pos_range),
        torch.linalg.vector_norm(position_a - position_d, dim=-1) <= dynamics.capture_R,
        getattr(dynamics, 'box_loses', False) & (position_a.abs().amax(-1) > dynamics.pos_range),
    ]
    event = torch.full((states.shape[0],), -1, dtype=torch.long, device=states.device)
    for index in reversed(range(len(happened))):   # the first listed event wins a tie
        event = torch.where(torch.as_tensor(happened[index], device=states.device), torch.full_like(event, index), event)
    return event


def play(starts, dynamics, defender, configs, args, generator):
    """Plays 2N games: rows 0..N-1 MPC vs MPC, rows N..2N-1 MPC attacker vs network defender."""
    count = starts.shape[0]
    steps = int(round(args.game_time / args.dt))
    states = torch.cat((starts, starts))
    batch = states.shape[0]
    trajectory = torch.zeros(batch, steps + 1, dynamics.state_dim, device=states.device)
    network_values = torch.zeros(batch, steps + 1, device=states.device)
    trajectory[:, 0] = states
    network_values[:, 0] = defender.values(states, None)
    event = torch.full((batch,), -1, dtype=torch.long, device=states.device)
    event_step = torch.full((batch,), steps, dtype=torch.long, device=states.device)
    plan_u = torch.zeros(batch, steps, dynamics.control_dim, device=states.device)
    plan_d = torch.zeros(batch, steps, dynamics.disturbance_dim, device=states.device)
    plan_start = 0
    for step in range(steps):
        remaining, offset = steps - step, step - plan_start
        if step % args.replan_every == 0:   # receding horizon: plans always reach the end of the game
            result = optimize_maxmin_sequences(
                states, torch.full((batch,), remaining * args.dt, device=states.device),
                plan_u[:, offset:offset + remaining], plan_d[:, offset:offset + remaining], None, dynamics,
                replace(configs['attacker'], horizon_steps=remaining),
                replace(configs['defender'], horizon_steps=remaining), generator)
            plan_u, plan_d, plan_start, offset = result.controls, result.defender_controls, step, 0
        u, d = plan_u[:, offset].clone(), plan_d[:, offset].clone()
        d[count:] = defender.query(states[count:], None).disturbances
        # games are played to the end (as for the library's labels); only the first event is recorded and drawn
        states = integrate_step(dynamics, states, u, d, args.dt, 'rk4')
        ended = event >= 0
        new_event = torch.where(ended, event, first_events(dynamics, states))
        event_step = torch.where(~ended & (new_event >= 0), torch.full_like(event_step, step + 1), event_step)
        event = new_event
        trajectory[:, step + 1] = states
        network_values[:, step + 1] = defender.values(states, None)
    outcome = reach_avoid_suffix_values(dynamics, trajectory)[:, 0]
    return trajectory.cpu(), network_values.cpu(), event.cpu(), event_step.cpu(), outcome.cpu()


def describe(event, event_step, dt, outcome):
    if event < 0:
        return 'time ran out, %s wins' % ('attacker' if outcome <= 0 else 'defender')
    name = EVENTS[event]
    return '%s at %.2f s (%s wins)' % (name, event_step * dt, 'attacker' if name in ATTACKER_WINS else 'defender')


def plot(path, trajectory, network_values, event, event_step, outcome, dynamics, args, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = trajectory.shape[0] // 2
    figure, axes = plt.subplots(3, count, figsize=(4.2 * count, 12.5), squeeze=False,
                                gridspec_kw=dict(height_ratios=[1, 1, 0.7]))
    limit = dynamics.pos_range
    for column in range(count):
        for row, label in ((0, 'MPC vs MPC'), (1, 'MPC vs network')):
            game = column + row * count
            axis = axes[row, column]
            end = int(event_step[game]) + 1
            path_a, path_d = trajectory[game, :end, [0, 2]], trajectory[game, :end, [4, 6]]
            axis.add_patch(plt.Circle((0, 0), dynamics.target_R, color='green', alpha=0.3))
            axis.add_patch(plt.Circle((0, 0), dynamics.defender_exclusion_R, color='gray', fill=False, ls='--'))
            axis.add_patch(plt.Rectangle((-limit, -limit), 2 * limit, 2 * limit, fill=False, color='black', lw=0.8))
            axis.plot(path_a[:, 0], path_a[:, 1], 'r-', lw=1.5, label='attacker (MPC)')
            axis.plot(path_d[:, 0], path_d[:, 1], 'b-', lw=1.5,
                      label='defender (%s)' % ('MPC' if row == 0 else 'network'))
            axis.plot(path_a[0, 0], path_a[0, 1], 'ro')
            axis.plot(path_d[0, 0], path_d[0, 1], 'bo')
            axis.plot(path_a[-1, 0], path_a[-1, 1], 'rx', ms=9, mew=2)
            axis.plot(path_d[-1, 0], path_d[-1, 1], 'bx', ms=9, mew=2)
            if int(event[game]) == EVENTS.index('captured'):
                axis.add_patch(plt.Circle(tuple(path_d[-1].tolist()), dynamics.capture_R, color='blue', fill=False))
            axis.set_title('scenario %d: %s\n%s\nV = %+.3f' % (
                column, label, describe(int(event[game]), int(event_step[game]), args.dt, float(outcome[game])),
                float(outcome[game])), fontsize=9)
            axis.set_xlim(-limit - 0.1, limit + 0.1)
            axis.set_ylim(-limit - 0.1, limit + 0.1)
            axis.set_aspect('equal')
            axis.grid(alpha=0.3)
            if column == 0:
                axis.legend(fontsize=7, loc='upper left')
        axis = axes[2, column]
        times = torch.arange(trajectory.shape[1]) * args.dt
        for row, style, label in ((0, '--', 'MPC vs MPC'), (1, '-', 'MPC vs network')):
            game = column + row * count
            end = int(event_step[game]) + 1
            axis.plot(times[:end], network_values[game, :end], style, color='black' if row else 'gray', label=label)
        axis.axhline(0, color='gray', lw=0.6)
        axis.set_xlabel('time [s]')
        axis.set_title('network V(%.1f s, x(t)) along the games' % args.time_to_go, fontsize=9)
        axis.grid(alpha=0.3)
        if column == 0:
            axis.set_ylabel('V (> 0: defender wins)')
            axis.legend(fontsize=7)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', default='./runs')
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--checkpoint', type=int, default=-1, help='-1 for model_final.pth, else the epoch number')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--start', type=float, nargs=8, action='append', default=None,
                        help='start state px_a vx_a py_a vy_a px_d vx_d py_d vy_d (repeat for several scenarios)')
    parser.add_argument('--num_scenarios', type=int, default=5, help='sampled starts when no --start is given')
    parser.add_argument('--scenario_seed', type=int, default=0)
    parser.add_argument('--game_time', type=float, default=None, help='game length (s), default the run\'s tMax')
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--setting', default='48x12', help='MPC setting as in utils/compare_mpc.py')
    parser.add_argument('--replan_every', type=int, default=5)
    parser.add_argument('--hold_steps', type=int, default=10)
    parser.add_argument('--noise_fraction', type=float, default=0.25)
    parser.add_argument('--chunk', type=int, default=48)
    parser.add_argument('--mpc_seed', type=int, default=0)
    parser.add_argument('--defender_policy', default='gradient', choices=['gradient', 'lookahead'])
    parser.add_argument('--lookahead_time', type=float, default=0.2)
    parser.add_argument('--time_to_go', type=float, default=None, help='fixed time-to-go of the network, default tMax')
    parser.add_argument('--safety_filter', type=lambda v: str(v).lower() in ('1', 'true', 'yes'), default=True)
    parser.add_argument('--out', default=None, help='image path (default: in the run folder)')
    args = parser.parse_args()

    device = torch.device(args.device)
    dynamics, model = load(args.experiments_dir, args.experiment_name, args.checkpoint, device)
    with open(os.path.join(args.experiments_dir, args.experiment_name, 'orig_opt.pickle'), 'rb') as f:
        run_tmax = float(pickle.load(f).tMax)
    args.game_time = args.game_time or run_tmax
    args.time_to_go = args.time_to_go or run_tmax
    defender = NetworkDefender(model, dynamics, device, policy=args.defender_policy, time_to_go=args.time_to_go,
                               lookahead_time=args.lookahead_time, dt=args.dt, safety_filter=args.safety_filter)

    if args.start:
        starts = torch.tensor(args.start, dtype=torch.float32)
    else:
        torch.manual_seed(args.scenario_seed)
        starts = sample_mpc_initial_states(dynamics, args.num_scenarios, 'engagement', defender_ring_width=1.0)
    starts = starts.to(device)

    setting = parse_setting(args.setting, 'rk4')
    options = SimpleNamespace(mpc_dt=args.dt, mpc_horizon_steps=int(round(args.game_time / args.dt)),
                              mpc_noise_std=None, mpc_noise_fraction=args.noise_fraction,
                              mpc_control_hold_steps=args.hold_steps, mpc_domain_constraint='none',
                              mpc_defender_keep_out=True)
    configs = make_configs(dynamics, options, setting, args.chunk)
    generator = torch.Generator(device=device).manual_seed(args.mpc_seed)
    print('playing %d scenarios x 2 games of %.2f s: MPC %s, network defender %s at time-to-go %.2f s%s' % (
        starts.shape[0], args.game_time, setting_name(setting), args.defender_policy, args.time_to_go,
        ' with safety filter' if args.safety_filter else ''), flush=True)
    trajectory, network_values, event, event_step, outcome = play(starts, dynamics, defender, configs, args, generator)

    count = starts.shape[0]
    for column in range(count):
        print('scenario %d  start %s' % (column, ' '.join('%+.2f' % v for v in starts[column].tolist())))
        print('    network V at start  %+.3f' % float(network_values[column, 0]))
        for row, label in ((0, 'MPC vs MPC    '), (1, 'MPC vs network')):
            game = column + row * count
            print('    %s  %s, V = %+.3f' % (label, describe(int(event[game]), int(event_step[game]), args.dt,
                                                         float(outcome[game])), float(outcome[game])))
    if args.safety_filter:
        print('safety filter changed the network defender\'s action in %d of %d steps' % (
            defender.safety_stats['changed'], defender.safety_stats['steps']))
    out = args.out or os.path.join(args.experiments_dir, args.experiment_name, 'mpc_vs_network_%s_%s%s.png' % (
        setting_name(setting), args.defender_policy, '_safe' if args.safety_filter else ''))
    plot(out, trajectory, network_values, event, event_step, outcome, dynamics, args,
         '%s, checkpoint %s: attacker MPC %s vs MPC defender (top) and vs network %s defender (middle)' % (
             args.experiment_name, args.checkpoint, setting_name(setting), args.defender_policy))
    print('figure -> %s' % out)


if __name__ == '__main__':
    main()
