"""Compare joint closed-loop MPC settings by the game values they produce, against a strong reference.

Every setting plays the same games the way training labels them: 'engagement' start states, game lengths
uniform in [--min_length, --max_length (default tMax)], closed-loop 'maxmin' MPC re-planning every mpc_replan_every steps, with no
network (zero warm start, terminal-set margin at the end). The dynamics and the remaining MPC options (dt,
horizon, noise, hold steps, domain constraint, keep-out, ring width, tMax) are taken from a training run.
The value of a game is the reach-avoid label of its start state, so short games test labels with little time
left and long games labels with a lot.

A setting is written SAMPLESxITERATIONS with optional suffixes -aK / -aKt (archive of K plans per player,
'threats' mode), -aKp (archive in 'picks' mode) and -euler / -rk4 (integrator; default --integrator),
e.g. 32x10, 32x10-a8, 32x10-a8p, 32x15-euler. Settings use the pursuit candidate and tie-break
(MPCConfig.pursuit, shown as -pur in the name); the suffix -nopur turns them off.

Reported per setting, against the reference V_ref = the mean of the reference's values over its
--reference_seeds seeds (averaging lowers the reference's own randomness):
  agree    share of games with the same winner (V <= 0: the attacker wins)
  bias     mean (V - V_ref); > 0: the setting's games end better for the defender than the reference's
  mae      mean |V - V_ref|
  >U       share of values above the straight-flight ceiling U (impossible for a perfect attacker)
  >U+.01   share of values more than 0.01 above U
  excess   mean (V - U) over the values above U
  sec      seconds to play all games
'noise' rows: each reference seed against the mean of the other reference seeds, and a setting's extra
seeds against its seed 0. Differences smaller than the reference's own noise are not measurable with it.

Cross-play (--cross_play SETTING ...): each player plans with its own solver (both solvers plan the whole game,
each executes only its own player; the executed game uses rk4).
  att_gap  V(setting attacker vs reference defender) - V_ref; > 0: the setting's attacker is weaker
  def_gap  V(reference attacker vs setting defender) - V_ref; < 0: the setting's defender is weaker

Each game set is cached in --out_dir, so adding settings later only plays the new ones. Example (Windows):
  .\\env\\Scripts\\python.exe -m utils.compare_mpc --experiment_name crazyflie_2d_mpc_2s_v5 ^
      --settings 32x5 32x10 32x15 64x5 64x10 64x15 --reference 64x30 --cross_play 32x15
"""

import argparse
import copy
import inspect
import os
import pickle
import time
from dataclasses import replace

import torch

from controllers.mpc import (MPCConfig, closed_loop_rollout, integrate_step, optimize_maxmin_sequences,
                             reach_avoid_suffix_values)
from dynamics import dynamics as dynamics_module
from utils.mpc_data import mpc_domain_constraint, sample_mpc_initial_states


def load_run(experiments_dir, experiment_name):
    with open(os.path.join(experiments_dir, experiment_name, 'orig_opt.pickle'), 'rb') as f:
        opt = pickle.load(f)
    dynamics_class = getattr(dynamics_module, opt.dynamics_class)
    dynamics = dynamics_class(**{
        name: getattr(opt, name)
        for name, param in inspect.signature(dynamics_class).parameters.items()
        if name != 'self' and (hasattr(opt, name) or param.default is inspect.Parameter.empty)})
    if getattr(dynamics, 'box_loses', False):   # as in training: MPC players may leave the box
        dynamics = copy.copy(dynamics)
        dynamics.box_loses = False
    return dynamics, opt


def parse_setting(text, default_integrator):
    parts = text.split('-')
    samples, iterations = (int(v) for v in parts[0].lower().split('x'))
    setting = dict(samples=samples, iterations=iterations, archive=0, archive_mode='threats',
                   integrator=default_integrator, pursuit=True)
    for part in parts[1:]:
        if part == 'nopur':
            setting['pursuit'] = False
        elif part in ('euler', 'rk4'):
            setting['integrator'] = part
        elif part.startswith('a'):
            mode = {'t': 'threats', 'p': 'picks'}.get(part[-1])
            setting['archive'] = int(part[1:-1] if mode else part[1:])
            setting['archive_mode'] = mode or 'threats'
        else:
            raise ValueError('unknown setting suffix %r in %r' % (part, text))
    return setting


def setting_name(setting):
    archive = '-a%d%s' % (setting['archive'], setting['archive_mode'][0]) if setting['archive'] else ''
    return '%dx%d%s-%s%s' % (setting['samples'], setting['iterations'], archive, setting['integrator'],
                             '-pur' if setting['pursuit'] else '')


def make_configs(dynamics, opt, setting, chunk):
    noise_std = getattr(opt, 'mpc_noise_std', None)
    configs = {}
    for player, bound, dim in (('attacker', dynamics.accel_max_a, dynamics.control_dim),
                               ('defender', dynamics.accel_max_d, dynamics.disturbance_dim)):
        configs[player] = MPCConfig(
            dt=opt.mpc_dt, horizon_steps=opt.mpc_horizon_steps,
            num_samples=setting['samples'], num_iterations=setting['iterations'],
            noise_std=noise_std if noise_std is not None else getattr(opt, 'mpc_noise_fraction', 0.25) * bound,
            control_lower=torch.full((dim,), -bound), control_upper=torch.full((dim,), bound),
            integration_method=setting['integrator'], candidate_chunk_size=chunk,
            control_hold_steps=getattr(opt, 'mpc_control_hold_steps', 10), archive_size=setting['archive'],
            archive_mode=setting['archive_mode'], pursuit=setting['pursuit'],
            **mpc_domain_constraint(dynamics, player, getattr(opt, 'mpc_domain_constraint', 'none'),
                                    getattr(opt, 'mpc_defender_keep_out', False)))
    return configs


def play(starts, times, dynamics, opt, attacker_setting, defender_setting, seed, chunk):
    """(values [B], trajectories [B, H+1, S]) of closed-loop games. With two different settings each one re-plans the whole game with its own
    solver and warm starts, and only its own player executes its plan."""
    device = starts.device
    horizon, dt, replan_every = opt.mpc_horizon_steps, opt.mpc_dt, opt.mpc_replan_every
    batch_size = starts.shape[0]
    zeros_u = torch.zeros(batch_size, horizon, dynamics.control_dim, device=device)
    zeros_d = torch.zeros(batch_size, horizon, dynamics.disturbance_dim, device=device)
    if attacker_setting == defender_setting:
        configs = make_configs(dynamics, opt, attacker_setting, chunk)
        generator = torch.Generator(device=device).manual_seed(seed)
        result = closed_loop_rollout(
            starts, times, zeros_u, zeros_d, None, dynamics, configs['attacker'], configs['defender'], 'joint',
            replan_every=replan_every, generator=generator, end_on_event=False, game_solver='maxmin')
        return result.score, result.states

    solvers = []
    for index, setting in enumerate((attacker_setting, defender_setting)):
        solvers.append(dict(configs=make_configs(dynamics, opt, setting, chunk), plans=(zeros_u, zeros_d),
                            generator=torch.Generator(device=device).manual_seed(seed + 7919 * index)))
    current_states, current_times = starts.clone(), times.clone()
    states = torch.zeros(batch_size, horizon + 1, dynamics.state_dim, device=device)
    states[:, 0] = current_states
    plan_start = 0
    for step in range(horizon):
        offset = step - plan_start
        if step % replan_every == 0:
            steps_left = horizon - step
            for solver in solvers:
                plan_u, plan_d = (plan[:, offset:offset + steps_left] for plan in solver['plans'])
                result = optimize_maxmin_sequences(
                    current_states, current_times, plan_u, plan_d, None, dynamics,
                    replace(solver['configs']['attacker'], horizon_steps=steps_left),
                    replace(solver['configs']['defender'], horizon_steps=steps_left), solver['generator'])
                solver['plans'] = (result.controls, result.defender_controls)
            plan_start, offset = step, 0
        u, d = solvers[0]['plans'][0][:, offset], solvers[1]['plans'][1][:, offset]
        next_states = integrate_step(dynamics, current_states, u, d, dt, 'rk4')
        current_states = torch.where((current_times > 0)[:, None], next_states, current_states)
        current_times = torch.clamp(current_times - dt, min=0.0)
        states[:, step + 1] = current_states
    return reach_avoid_suffix_values(dynamics, states)[:, 0], states


def cached_values(path, compute):
    if os.path.exists(path):
        return torch.load(path)
    start = time.perf_counter()
    values, states = compute()
    result = {'values': values.cpu(), 'states': states.cpu(), 'seconds': time.perf_counter() - start}
    torch.save(result, path)
    return result


def plot_games(path, game, runs, times, ceiling, dynamics, dt):
    """One panel per run: attacker (red) and defender (blue) paths of one game, x = first capture,
    * = first target hit, o = start."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = min(4, len(runs))
    rows = (len(runs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    steps = int(round(float(times[game]) / dt)) + 1
    limit = float(dynamics.state_var[0]) if hasattr(dynamics, 'state_var') else 2.0
    for axis, (label, run) in zip(axes.flat, runs):
        states = run['states'][game, :steps]
        attacker, defender = states[:, [0, 2]], states[:, [4, 6]]
        axis.add_patch(plt.Circle((0, 0), dynamics.target_R, color='green', alpha=0.3))
        axis.add_patch(plt.Circle((0, 0), dynamics.defender_exclusion_R, color='gray', fill=False, linestyle='--'))
        axis.plot(attacker[:, 0], attacker[:, 1], 'r-', lw=1.5, label='attacker')
        axis.plot(defender[:, 0], defender[:, 1], 'b-', lw=1.5, label='defender')
        axis.plot(attacker[0, 0], attacker[0, 1], 'ro')
        axis.plot(defender[0, 0], defender[0, 1], 'bo')
        captured = torch.nonzero((attacker - defender).norm(dim=-1) <= dynamics.capture_R)
        if len(captured):
            k = int(captured[0])
            axis.plot(attacker[k, 0], attacker[k, 1], 'kx', ms=10, mew=2)
            axis.add_patch(plt.Circle(tuple(defender[k].tolist()), dynamics.capture_R, color='blue', fill=False, lw=0.8))
        hit = torch.nonzero(attacker.norm(dim=-1) <= dynamics.target_R)
        if len(hit):
            k = int(hit[0])
            axis.plot(attacker[k, 0], attacker[k, 1], 'k*', ms=12)
        axis.set_title('%s\nV = %+.3f  (U = %+.3f)' % (label, float(run['values'][game]), float(ceiling[game])),
                       fontsize=9)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect('equal')
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[len(runs):]:
        axis.axis('off')
    axes.flat[0].legend(fontsize=8, loc='upper left')
    figure.suptitle('game %d, %.2f s' % (game, float(times[game])))
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', default='./runs')
    parser.add_argument('--experiment_name', required=True, help='training run whose dynamics and MPC options are used')
    parser.add_argument('--settings', nargs='+', required=True, help='e.g. 32x5 32x10 64x15-a8 32x15-euler')
    parser.add_argument('--reference', default='64x30', help='strong setting the others are compared with')
    parser.add_argument('--reference_seeds', type=int, default=2, help='>= 2 to measure the reference noise level')
    parser.add_argument('--setting_seeds', type=int, default=1, help='>= 2 to also report each setting\'s own noise')
    parser.add_argument('--cross_play', nargs='*', default=[], help='settings to cross-play against the reference')
    parser.add_argument('--plot', type=int, default=0,
                        help='plot this many games (those where the plotted runs disagree most) into OUT_DIR/plots')
    parser.add_argument('--plot_games', type=int, nargs='*', default=[], help='game indices to plot (added to --plot)')
    parser.add_argument('--integrator', default='rk4', choices=['euler', 'rk4'], help='default integrator of settings')
    parser.add_argument('--num_games', type=int, default=128)
    parser.add_argument('--min_length', type=float, default=0.1, help='game lengths uniform in [min_length, max_length] (s)')
    parser.add_argument('--max_length', type=float, default=None,
                        help='longest game (s), default the run\'s tMax; the MPC horizon is extended to cover it')
    parser.add_argument('--state_seed', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=None, help='attacker plans per rollout chunk (memory)')
    parser.add_argument('--out_dir', default='mpc_compare')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    dynamics, opt = load_run(args.experiments_dir, args.experiment_name)
    max_length = opt.tMax if args.max_length is None else args.max_length
    if args.min_length > max_length:
        raise ValueError('--min_length must not exceed the longest game')
    # the plans must reach the end of the longest game (no network is used, so tMax itself does not matter)
    opt = copy.copy(opt)
    opt.mpc_horizon_steps = max(opt.mpc_horizon_steps, int(round(max_length / opt.mpc_dt)))

    # the start states fix everything the cached games depend on besides the setting and seed
    game_key = dict(experiment=args.experiment_name, num_games=args.num_games, min_length=args.min_length,
                    state_seed=args.state_seed, max_length=max_length)
    os.makedirs(args.out_dir, exist_ok=True)
    starts_path = os.path.join(args.out_dir, 'starts.pt')
    if os.path.exists(starts_path):
        saved = torch.load(starts_path)
        if saved['key'] != game_key:
            raise ValueError('%s holds games for %s; use another --out_dir' % (args.out_dir, saved['key']))
    else:
        torch.manual_seed(args.state_seed)
        starts = sample_mpc_initial_states(dynamics, args.num_games, 'engagement',
                                           defender_ring_width=getattr(opt, 'mpc_defender_ring_width', 1.0))
        steps = torch.randint(int(round(args.min_length / opt.mpc_dt)), int(round(max_length / opt.mpc_dt)) + 1,
                              (args.num_games,))
        saved = {'key': game_key, 'states': starts, 'times': steps * opt.mpc_dt}
        torch.save(saved, starts_path)
    starts, times = saved['states'].to(device), saved['times'].to(device)
    ceiling = dynamics.value_upper_bound(times.cpu(), starts.cpu(), opt.mpc_dt)

    def games(attacker, defender, seed):
        name = setting_name(attacker) + ('' if attacker == defender else '_vs_' + setting_name(defender))
        path = os.path.join(args.out_dir, '%s_seed%d.pt' % (name, seed))
        print('playing' if not os.path.exists(path) else 'cached ', name, 'seed', seed, flush=True)
        return cached_values(path, lambda: play(starts, times, dynamics, opt, attacker, defender, seed, args.chunk))

    reference = parse_setting(args.reference, args.integrator)
    reference_runs = [games(reference, reference, seed) for seed in range(args.reference_seeds)]
    plotted = [(setting_name(reference) + ' seed 0', reference_runs[0])]
    reference_values = torch.stack([run['values'] for run in reference_runs])
    ref = reference_values.mean(dim=0)

    def row(label, values, seconds, against=ref):
        diff = values - against
        agree = ((values <= 0) == (against <= 0)).float().mean()
        excess = values - ceiling
        above = excess > 1e-4
        mean_excess = excess[above].mean().item() if above.any() else 0.0
        print('%-26s %6.3f %+7.3f %6.3f %6.3f %7.3f %7.3f %8.0f' % (
            label, agree, diff.mean(), diff.abs().mean(), above.float().mean(), (excess > 0.01).float().mean(),
            mean_excess, seconds))

    print('\n%d games, lengths %.1f-%.1f s, reference %s (mean of %d seeds); attacker wins in %.2f of reference games'
          % (args.num_games, args.min_length, max_length, setting_name(reference), len(reference_runs),
             (ref <= 0).float().mean()))
    print('%-26s %6s %7s %6s %6s %7s %7s %8s' % ('setting', 'agree', 'bias', 'mae', '>U', '>U+.01', 'excess', 'sec'))
    row(setting_name(reference) + ' (mean)', ref, sum(run['seconds'] for run in reference_runs))
    if len(reference_runs) > 1:
        for seed, run in enumerate(reference_runs):
            others = torch.cat((reference_values[:seed], reference_values[seed + 1:])).mean(dim=0)
            row('  noise: ref seed %d vs rest' % seed, run['values'], run['seconds'], against=others)
    for text in args.settings:
        setting = parse_setting(text, args.integrator)
        runs = [games(setting, setting, seed) for seed in range(args.setting_seeds)]
        row(setting_name(setting), runs[0]['values'], runs[0]['seconds'])
        plotted.append((setting_name(setting), runs[0]))
        for seed, run in enumerate(runs[1:], start=1):
            row('  noise: seed %d vs seed 0' % seed, run['values'], run['seconds'], against=runs[0]['values'])

    if args.cross_play:
        print('\ncross-play against %s (att_gap > 0: weaker attacker, def_gap < 0: weaker defender)' %
              setting_name(reference))
        print('%-26s %8s %8s %8s %8s' % ('setting', 'att_gap', 'agree', 'def_gap', 'agree'))
        for text in args.cross_play:
            setting = parse_setting(text, args.integrator)
            attacker_run, defender_run = games(setting, reference, 0), games(reference, setting, 0)
            plotted += [('attacker %s vs ref' % setting_name(setting), attacker_run),
                        ('ref vs defender %s' % setting_name(setting), defender_run)]
            attacker_side, defender_side = attacker_run['values'], defender_run['values']
            print('%-26s %+8.3f %8.3f %+8.3f %8.3f' % (
                setting_name(setting), (attacker_side - ref).mean(), ((attacker_side <= 0) == (ref <= 0)).float().mean(),
                (defender_side - ref).mean(), ((defender_side <= 0) == (ref <= 0)).float().mean()))

    if args.plot or args.plot_games:
        missing = [label for label, run in plotted if 'states' not in run]
        if missing:
            print('\nno trajectories saved for %s (played before trajectories were stored); delete their files '
                  'in %s to replay them' % (', '.join(missing), args.out_dir))
        plotted = [(label, run) for label, run in plotted if 'states' in run]
        if plotted:
            spread = torch.stack([run['values'] for _, run in plotted])
            spread = spread.amax(dim=0) - spread.amin(dim=0)
            chosen = list(dict.fromkeys(args.plot_games + spread.argsort(descending=True)[:args.plot].tolist()))
            plot_dir = os.path.join(args.out_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            for game in chosen:
                plot_games(os.path.join(plot_dir, 'game_%03d.png' % game), game, plotted, times.cpu(), ceiling,
                           dynamics, opt.mpc_dt)
            print('\nplotted games %s (--plot_games first, then the largest value spread) into %s' % (chosen, plot_dir))


if __name__ == '__main__':
    main()
