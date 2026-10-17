"""Plot games stored in an MPC library chunk (or its holdout.pt), up to their first event.

A chunk stores, game by game in time order, the labelled states before each game's first capture / target hit /
breach, plus each game's start state; the games are recovered by finding the start states in that sequence.
Each panel shows the attacker (red) and defender (blue) paths from the start (o) to the last stored state (x: the first event, or the
end of a game that ran out of time),
coloured dots along the attacker path give the label (MPC game value) at that state (blue: defender wins,
red: attacker wins). The title gives the game length, the start label and the number of stored states.

  .\\env\\Scripts\\python.exe -m utils.view_library --library_dir mpc_library_3s --chunk 3 --games 12
  .\\env\\Scripts\\python.exe -m utils.view_library --library_dir mpc_library_3s --file holdout.pt --games 12
"""

import argparse
import os

import torch


def split_games(chunk):
    """[(start index, end index)] of each game's stored states in the chunk's flat sequence (None if missing)."""
    states, starts = chunk['states'], chunk['start_states']
    spans, position = [], 0
    for start in starts:
        matches = torch.nonzero((states[position:] == start).all(dim=-1)).flatten()
        if not len(matches):
            spans.append(None)
            continue
        first = position + int(matches[0])
        spans.append(first)
        position = first + 1
    found = [s for s in spans if s is not None] + [states.shape[0]]
    ends = {begin: found[i + 1] for i, begin in enumerate(found[:-1])}
    return [None if s is None else (s, ends[s]) for s in spans]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library_dir', required=True)
    parser.add_argument('--chunk', type=int, default=0, help='chunk index (ignored with --file)')
    parser.add_argument('--file', default=None, help='a file in the library folder instead, e.g. holdout.pt')
    parser.add_argument('--games', type=int, default=12, help='number of games to plot')
    parser.add_argument('--first_game', type=int, default=0)
    parser.add_argument('--out', default=None, help='image path (default: <library_dir>/plots/<file>.png)')
    args = parser.parse_args()

    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    name = args.file or 'chunk_%05d.pt' % args.chunk
    chunk = torch.load(os.path.join(args.library_dir, name), map_location='cpu')
    with open(os.path.join(args.library_dir, 'meta.json')) as f:
        game = json.load(f)['game']
    spans = split_games(chunk)
    chosen = list(range(args.first_game, min(args.first_game + args.games, len(spans))))

    columns = min(4, len(chosen))
    rows = (len(chosen) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.2 * rows), squeeze=False)
    for axis, index in zip(axes.flat, chosen):
        axis.add_patch(plt.Circle((0, 0), game['target_R'], color='green', alpha=0.3))
        axis.add_patch(plt.Circle((0, 0), game['defender_exclusion_R'], color='gray', fill=False, linestyle='--'))
        span = spans[index]
        if span is None:
            axis.set_title('game %d: start not stored' % index, fontsize=9)
        else:
            states, values = chunk['states'][span[0]:span[1]], chunk['values'][span[0]:span[1]]
            attacker, defender = states[:, [0, 2]], states[:, [4, 6]]
            axis.plot(attacker[:, 0], attacker[:, 1], 'r-', lw=1, alpha=0.6)
            axis.plot(defender[:, 0], defender[:, 1], 'b-', lw=1.5)
            axis.scatter(attacker[:, 0], attacker[:, 1], c=values, cmap='coolwarm_r', vmin=-0.25, vmax=0.25, s=8,
                         zorder=3)
            axis.plot(attacker[0, 0], attacker[0, 1], 'ro')
            axis.plot(defender[0, 0], defender[0, 1], 'bo')
            axis.plot(attacker[-1, 0], attacker[-1, 1], 'rx', ms=8)
            axis.plot(defender[-1, 0], defender[-1, 1], 'bx', ms=8)
            axis.add_patch(plt.Circle(tuple(defender[-1].tolist()), game['capture_R'], color='blue', fill=False,
                                      lw=0.6))
            start_value = float(chunk['start_values'][index])
            axis.set_title('game %d: %.2f s, V = %+.3f (%s)\n%d states stored (until the first event or the end)' % (
                index, float(chunk['start_times'][index]), start_value,
                'attacker wins' if start_value <= 0 else 'defender wins', span[1] - span[0]), fontsize=9)
        axis.set_xlim(-2, 2)
        axis.set_ylim(-2, 2)
        axis.set_aspect('equal')
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[len(chosen):]:
        axis.axis('off')
    figure.suptitle('%s / %s (dots on the attacker path: label at that state, blue = defender wins)' % (
        args.library_dir, name), fontsize=10)
    figure.tight_layout()
    out = args.out or os.path.join(args.library_dir, 'plots', name.replace('.pt', '.png'))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    figure.savefig(out, dpi=110)
    print('%d games plotted -> %s' % (len(chosen), out))


if __name__ == '__main__':
    main()
