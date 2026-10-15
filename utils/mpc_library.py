"""Build a fixed library of MPC-labelled game states once, then train on it without refreshes.

The labels do not depend on the network (joint 'maxmin' MPC without the network), so they can be generated
before training and reused by every run with the same game. Games follow the training refresh rules: closed-loop
MPC re-planning every mpc_replan_every steps, 'engagement' starts, full games, only the states before the first
capture / target hit / breach, cropped to the domain (if the run does), labels capped at the exact ceiling.
Half of the games (--full_length_share) last the full tMax; the others have lengths uniform in
[--min_length, tMax], so every time-to-go has labels for the early curriculum.

The game and the remaining MPC options come from a base training run, changed with --override NAME=VALUE
(e.g. tMax=3 vel_max_a=3.0 vel_max_d=3.0). The MPC setting is written as in utils/compare_mpc.py (64x10,
32x15-a8, ... with the pursuit candidate unless -nopur). The horizon is extended to cover tMax.

Layout of --library_dir:
  meta.json          the game, MPC and game-length settings; a later build into the same folder must match
  chunk_00000.pt ... one batch of --games_per_chunk games each (times, states, values; start_* per game)
  holdout.pt         --holdout_games games of the full tMax (separate seeds), used as the run's held-out set
Each chunk has its own seed (index + 1000), so the build can be stopped and restarted (finished chunks are
kept), and several terminals can build into one folder with --workers N --worker K (K = 0 .. N-1).

Example (Windows):
  .\\env\\Scripts\\python.exe -m utils.mpc_library --experiment_name crazyflie_2d_mpc_2s_v5 ^
      --library_dir mpc_library_3s --override tMax=3 vel_max_a=3.0 vel_max_d=3.0 --setting 64x10 --num_chunks 150
Training: add --mpc_library_dir mpc_library_3s (refreshes are then off and the held-out set comes from it).
"""

import argparse
import copy
import inspect
import json
import math
import os
import time

import torch

from controllers.mpc import closed_loop_rollout
from utils.compare_mpc import build_dynamics, load_options, make_configs, mpc_dynamics, parse_setting, setting_name
from utils.mpc_data import cap_to_ceiling, mpc_label_mask, sample_mpc_initial_states

CHUNK_SEED_OFFSET = 1000
HOLDOUT_SEED_OFFSET = 10 ** 6


def apply_overrides(opt, overrides):
    """Copy of the run options with NAME=VALUE overrides, converted to the type of the existing value."""
    opt = copy.copy(opt)
    for item in overrides:
        name, _, text = item.partition('=')
        if not hasattr(opt, name):
            raise ValueError('unknown option %r in --override' % name)
        old = getattr(opt, name)
        if text.lower() == 'none':
            value = None
        elif isinstance(old, bool):
            value = text.lower() in ('1', 'true', 'yes')
        elif isinstance(old, int):
            value = int(text)
        else:
            value = float(text) if old is None or isinstance(old, float) else type(old)(text)
        setattr(opt, name, value)
    return opt


def game_parameters(dynamics):
    """The dynamics constructor arguments (the game rules), as plain values."""
    return {name: getattr(dynamics, name) for name in inspect.signature(type(dynamics)).parameters
            if name != 'self' and hasattr(dynamics, name)}


def play_and_label(starts, times, game, players, opt, setting, seed, chunk):
    """Closed-loop MPC games from `starts` with time-to-go `times`, labelled like a training refresh."""
    device = starts.device
    horizon = opt.mpc_horizon_steps
    zeros_u = torch.zeros(starts.shape[0], horizon, players.control_dim, device=device)
    zeros_d = torch.zeros(starts.shape[0], horizon, players.disturbance_dim, device=device)
    configs = make_configs(players, opt, setting, chunk)
    result = closed_loop_rollout(
        starts, times, zeros_u, zeros_d, None, players, configs['attacker'], configs['defender'], 'joint',
        replan_every=opt.mpc_replan_every, generator=torch.Generator(device=device).manual_seed(seed),
        end_on_event=False, game_solver='maxmin')
    label_times, keep = mpc_label_mask(players, result, times, opt.mpc_dt, drop_after_event=True,
                                       crop_to_domain=getattr(opt, 'mpc_crop_to_domain', True))
    games = {
        'times': label_times[keep].cpu(), 'states': result.states[keep].cpu(), 'values': result.suffix_values[keep].cpu(),
        'start_times': label_times[:, 0].cpu(), 'start_states': result.states[:, 0].cpu(),
        'start_values': result.suffix_values[:, 0].cpu(),
    }
    capped = {}
    for prefix in ('', 'start_'):   # the ceiling of the network's game (box rule included), as in training
        games[prefix + 'values'], excess = cap_to_ceiling(
            game, games[prefix + 'times'], games[prefix + 'states'], games[prefix + 'values'], opt.mpc_dt)
        capped[prefix] = (excess > 1e-3).float().mean().item() if excess.numel() else 0.0
    games['capped_share'] = capped['']
    return games


def sample_starts(players, opt, count, seed, lengths):
    """Engagement starts (global RNG seeded with `seed`) and their game lengths."""
    torch.manual_seed(seed)
    starts = sample_mpc_initial_states(players, count, 'engagement',
                                       defender_ring_width=getattr(opt, 'mpc_defender_ring_width', 1.0))
    if lengths == 'full':
        steps = torch.full((count,), int(round(opt.tMax / opt.mpc_dt)))
    else:
        full_count, min_steps = lengths
        steps = torch.randint(min_steps, int(round(opt.tMax / opt.mpc_dt)) + 1, (count,))
        steps[:full_count] = int(round(opt.tMax / opt.mpc_dt))
    return starts, steps.float() * opt.mpc_dt


def atomic_save(value, path):
    temporary = path + '.tmp'
    torch.save(value, temporary)
    os.replace(temporary, path)


def build(args):
    device = torch.device(args.device)
    opt = apply_overrides(load_options(args.experiments_dir, args.experiment_name), args.override)
    opt.mpc_horizon_steps = max(opt.mpc_horizon_steps, int(math.ceil(opt.tMax / opt.mpc_dt - 1e-6)))
    game = build_dynamics(opt)
    players = mpc_dynamics(game)
    setting = parse_setting(args.setting, args.integrator)
    meta = {
        'game': game_parameters(game), 'tMax': opt.tMax, 'dt': opt.mpc_dt,
        'mpc': dict(setting=setting_name(setting), horizon_steps=opt.mpc_horizon_steps,
                    replan_every=opt.mpc_replan_every, hold_steps=getattr(opt, 'mpc_control_hold_steps', 10),
                    noise_std=getattr(opt, 'mpc_noise_std', None), noise_fraction=getattr(opt, 'mpc_noise_fraction', 0.25),
                    domain_constraint=getattr(opt, 'mpc_domain_constraint', 'none'),
                    defender_keep_out=getattr(opt, 'mpc_defender_keep_out', False)),
        'games': dict(per_chunk=args.games_per_chunk, full_length_share=args.full_length_share,
                      min_length=args.min_length, starts='engagement',
                      defender_ring_width=getattr(opt, 'mpc_defender_ring_width', 1.0),
                      crop_to_domain=getattr(opt, 'mpc_crop_to_domain', True), drop_after_event=True, capped=True),
        'base_experiment': args.experiment_name, 'overrides': args.override, 'holdout_games': args.holdout_games,
    }
    os.makedirs(args.library_dir, exist_ok=True)
    meta_path = os.path.join(args.library_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            saved = json.load(f)
        if json.loads(json.dumps(meta)) != saved:
            raise ValueError('%s was built with other settings:\n%s\nuse another --library_dir' % (
                args.library_dir, json.dumps(saved, indent=1)))
    else:
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=1)
    print('library %s: game %s, tMax %.2f s, MPC %s, horizon %d steps' % (
        args.library_dir, meta['game'], opt.tMax, setting_name(setting), opt.mpc_horizon_steps), flush=True)

    holdout_path = os.path.join(args.library_dir, 'holdout.pt')
    if args.worker == 0 and args.holdout_games > 0 and not os.path.exists(holdout_path):
        parts, start = [], time.perf_counter()
        for batch, first in enumerate(range(0, args.holdout_games, args.games_per_chunk)):
            count = min(args.games_per_chunk, args.holdout_games - first)
            starts, times = sample_starts(players, opt, count, HOLDOUT_SEED_OFFSET + batch, 'full')
            parts.append(play_and_label(starts.to(device), times.to(device), game, players, opt, setting,
                                        HOLDOUT_SEED_OFFSET + batch, args.chunk))
            print('  held-out games %d/%d (%.0f s)' % (first + count, args.holdout_games,
                                                       time.perf_counter() - start), flush=True)
        holdout = {key: torch.cat([part[key] for part in parts]) for key in
                   ('times', 'states', 'values', 'start_times', 'start_states', 'start_values')}
        atomic_save(holdout, holdout_path)
        print('held-out set: %d games, %d labelled states -> %s' % (
            holdout['start_values'].numel(), holdout['values'].numel(), holdout_path), flush=True)

    full_count = int(round(args.full_length_share * args.games_per_chunk))
    min_steps = max(1, int(round(args.min_length / opt.mpc_dt)))
    todo = [index for index in range(args.num_chunks) if index % args.workers == args.worker
            and not os.path.exists(os.path.join(args.library_dir, 'chunk_%05d.pt' % index))]
    print('%d chunks to play (worker %d of %d)' % (len(todo), args.worker, args.workers), flush=True)
    start = time.perf_counter()
    for done, index in enumerate(todo, start=1):
        seed = CHUNK_SEED_OFFSET + index
        starts, times = sample_starts(players, opt, args.games_per_chunk, seed, (full_count, min_steps))
        chunk_start = time.perf_counter()
        games = play_and_label(starts.to(device), times.to(device), game, players, opt, setting, seed, args.chunk)
        atomic_save(games, os.path.join(args.library_dir, 'chunk_%05d.pt' % index))
        elapsed = time.perf_counter() - start
        print('chunk %05d: %d labels, %.1f%% capped, %.0f s; %d/%d done, about %.1f h left' % (
            index, games['values'].numel(), 100 * games['capped_share'], time.perf_counter() - chunk_start,
            done, len(todo), elapsed / done * (len(todo) - done) / 3600), flush=True)
    chunks = sorted(name for name in os.listdir(args.library_dir) if name.startswith('chunk_') and name.endswith('.pt'))
    print('library holds %d chunks' % len(chunks))


def load_library(library_dir):
    """All labels of a library (CPU tensors), its meta data and its held-out set (or None)."""
    with open(os.path.join(library_dir, 'meta.json')) as f:
        meta = json.load(f)
    names = sorted(name for name in os.listdir(library_dir) if name.startswith('chunk_') and name.endswith('.pt'))
    if not names:
        raise ValueError('%s holds no chunks yet' % library_dir)
    chunks = [torch.load(os.path.join(library_dir, name), map_location='cpu') for name in names]
    holdout_path = os.path.join(library_dir, 'holdout.pt')
    holdout = torch.load(holdout_path, map_location='cpu') if os.path.exists(holdout_path) else None
    return {'meta': meta, 'chunks': len(names), 'holdout': holdout,
            **{key: torch.cat([chunk[key] for chunk in chunks]) for key in ('times', 'states', 'values')}}


def check_library_matches(library, dynamics, tMax):
    """Raise if the library was built for another game than the run's dynamics, or does not cover tMax."""
    built = library['meta']['game']
    run = json.loads(json.dumps(game_parameters(dynamics)))
    different = {name: (built.get(name), value) for name, value in run.items()
                 if not (built.get(name) == value or (isinstance(value, float) and isinstance(built.get(name), (int, float))
                                                       and abs(built[name] - value) < 1e-9))}
    if different:
        raise ValueError('MPC library game differs from this run (library, run): %s' % different)
    if tMax > library['meta']['tMax'] + 1e-9:
        raise ValueError('MPC library covers tMax %.2f s, the run needs %.2f s' % (library['meta']['tMax'], tMax))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--experiments_dir', default='./runs')
    parser.add_argument('--experiment_name', required=True, help='base run: game and MPC options')
    parser.add_argument('--override', nargs='*', default=[], help='NAME=VALUE changes to the base run options')
    parser.add_argument('--library_dir', required=True)
    parser.add_argument('--setting', default='64x10', help='MPC setting, as in utils/compare_mpc.py')
    parser.add_argument('--integrator', default='rk4', choices=['euler', 'rk4'])
    parser.add_argument('--num_chunks', type=int, default=150, help='total chunks in the library')
    parser.add_argument('--games_per_chunk', type=int, default=128)
    parser.add_argument('--full_length_share', type=float, default=0.5, help='share of games lasting the full tMax')
    parser.add_argument('--min_length', type=float, default=0.1, help='shortest game of the others (s)')
    parser.add_argument('--holdout_games', type=int, default=512)
    parser.add_argument('--chunk', type=int, default=32, help='attacker plans per rollout batch (GPU memory)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        parser.error('--worker must be in 0 .. --workers - 1')
    build(args)


if __name__ == '__main__':
    main()
