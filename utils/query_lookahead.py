"""Query the trained network at one state and show the lookahead defender's 9 candidate values.

For each defender acceleration (4 box corners, 4 axis directions, none) the state is predicted
--lookahead_time ahead (attacker keeps its velocity, speed limits applied) and V is evaluated there.
Also prints V at the state itself and the gradient (bang-bang) defender's choice for comparison.

Run from the SmartDrones directory, state order [px_a, vx_a, py_a, vy_a, px_d, vx_d, py_d, vy_d]:
  python -m utils.query_lookahead --experiment_name crazyflie_2d_mpc_2s_v5 --checkpoint 200000 \
      --state 1.5 0 -0.2 0 0 0 0.7 0
"""

import argparse
import inspect
import os
import pickle

import torch

from controllers.bang_bang import NeuralBangBangController
from controllers.mpc import integrate_step
from dynamics import dynamics as dynamics_module
from utils import modules


def load(experiments_dir, experiment_name, checkpoint, device):
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'rb') as f:
        orig_opt = pickle.load(f)
    dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
    dynamics = dynamics_class(**{
        name: getattr(orig_opt, name)
        for name, param in inspect.signature(dynamics_class).parameters.items()
        if name != 'self' and (hasattr(orig_opt, name) or param.default is inspect.Parameter.empty)})
    dynamics.deepreach_model = orig_opt.deepreach_model
    model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type=orig_opt.model,
                                 mode=orig_opt.model_mode, final_layer_factor=1.,
                                 hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
    checkpoints_dir = os.path.join(experiment_dir, 'training', 'checkpoints')
    if checkpoint == -1:
        model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model_final.pth'), map_location=device))
    else:
        path = os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % checkpoint)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
    return dynamics, model.to(device).eval()


def values(model, dynamics, states, times):
    coords = dynamics.coord_to_input(torch.cat((times.unsqueeze(-1), states), dim=-1))
    with torch.no_grad():
        results = model({'coords': coords})
        return dynamics.io_to_value(results['model_in'], results['model_out'].squeeze(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', default='./runs')
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--checkpoint', type=int, default=-1, help='-1 for model_final.pth, else epoch number')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--state', type=float, nargs=8, required=True,
                        help='px_a vx_a py_a vy_a px_d vx_d py_d vy_d (m, m/s)')
    parser.add_argument('--time_to_go', type=float, default=2.0, help='time-to-go of the query (s)')
    parser.add_argument('--lookahead_time', type=float, default=0.2)
    parser.add_argument('--dt', type=float, default=0.02, help='integration step of the prediction (s)')
    parser.add_argument('--countdown', action='store_true',
                        help='evaluate predicted states at time_to_go - lookahead_time instead of time_to_go')
    args = parser.parse_args()

    device = torch.device(args.device)
    dynamics, model = load(args.experiments_dir, args.experiment_name, args.checkpoint, device)
    state = torch.tensor([args.state], dtype=torch.float32, device=device)
    now = torch.tensor([args.time_to_go], device=device)

    query = NeuralBangBangController(model, dynamics, device).query(state, now)
    gradient = query.gradients[0]
    print('V(%.2f, state) = %+.4f' % (args.time_to_go, query.values[0].item()))
    print('dV/d(defender velocity) = (%+.4f, %+.4f)   dV/d(defender position) = (%+.4f, %+.4f)' % (
        gradient[5].item(), gradient[7].item(), gradient[4].item(), gradient[6].item()))
    print('gradient (bang-bang) defender acceleration = (%+.1f, %+.1f)' % tuple(query.disturbances[0].tolist()))

    a = dynamics.accel_max_d
    candidates = torch.tensor([[a, a], [a, -a], [-a, a], [-a, -a], [a, 0.0], [-a, 0.0], [0.0, a], [0.0, -a],
                               [0.0, 0.0]], dtype=torch.float32, device=device)
    steps = max(int(round(args.lookahead_time / args.dt)), 1)
    step_dt = args.lookahead_time / steps
    predicted = state.repeat(len(candidates), 1)
    attacker = torch.zeros_like(candidates)                 # attacker keeps its velocity
    for _ in range(steps):
        predicted = integrate_step(dynamics, predicted, attacker, candidates, step_dt, 'euler')
    eval_time = args.time_to_go - args.lookahead_time if args.countdown else args.time_to_go
    candidate_values = values(model, dynamics, predicted, torch.full((len(candidates),), max(eval_time, 0.0), device=device))
    best = candidate_values.argmax().item()

    print('\nlookahead %.2f s, predicted states scored at time-to-go %.2f s:' % (args.lookahead_time, eval_time))
    print('  accel (x, y)    defender pos      defender vel        V        V - best')
    for i in range(len(candidates)):
        d = predicted[i]
        print('  (%+4.1f, %+4.1f)  (%+.3f, %+.3f)  (%+.3f, %+.3f)  %+.4f  %+.4f%s' % (
            candidates[i, 0].item(), candidates[i, 1].item(), d[4].item(), d[6].item(), d[5].item(), d[7].item(),
            candidate_values[i].item(), (candidate_values[i] - candidate_values[best]).item(),
            '   <- chosen' if i == best else ''))
    print('spread (max - min) = %.4f' % (candidate_values.max() - candidate_values.min()).item())


if __name__ == '__main__':
    main()
