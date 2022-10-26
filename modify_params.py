#!/usr/bin/env python3

import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid

from space_conf_modify import space, init
print(space)


def main(args):
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    args.n = int(args.n)
    fn = os.path.join(out_dir, 'modified_params')
    ps = []

    for i in range(args.n):

        new_params = {}
        change_idx = np.random.randint(0, len(space))
        change_key = list(space.keys())[change_idx]
        change_vals = space[change_key].copy()
        change_vals.remove(init[change_key])
        new_val = np.random.choice(change_vals)
        new_params[change_key] = new_val

        for j in range(len(space)):
            if j != change_idx:
                if 0.3 < np.random.rand():
                    key = list(space.keys())[j]
                    new_params[key] = init[key]
                else:
                    key = list(space.keys())[j]
                    new_params[key] = np.random.choice(space[key])
        ps.append(new_params)

    with open(fn, 'w') as fp:
        for p in ps:
            p_str = ' '.join([args.format.format(name=k, value=v) for k, v in p.items()])
            if args.extra:
                p_str = args.extra + ' ' + p_str
            fp.write('sbatch --gres=gpu:rtx8000:1 --mem=48G run.sh ImageCoDe.py ' + p_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a set of hyper parameters')
    parser.add_argument('output', type=str,
                        help='output directory')
    parser.add_argument('n', type=str,
                        help='random search: number of hyper parameter sets '
                        'to sample, for grid search: set to "all"')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for deterministic runs')
    parser.add_argument('--format', type=str, default='--{name}={value}',
                        help='format for parameter arguments, default is '
                        '--{name}={value}')
    parser.add_argument('--extra', type=str, help='Extra arguments to add')

    args = parser.parse_args()
    main(args)