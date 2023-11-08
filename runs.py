import subprocess
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--algo', type=str)
parser.add_argument('--sigma_max', type=float)
parser.add_argument('--sing_val_scale', type=float, default=2)
parser.add_argument('--sing_val_init', type=float, default=1)
parser.add_argument('--error_delta', type=float, default=.1)

args = parser.parse_args()

alg = args.algo
sigma_max = args.sigma_max
svs = args.sing_val_scale
svi = args.sing_val_init
err = args.error_delta

bs = 'python train.py'
run = lambda x: f'--run {x} --algo {alg} --sigma_max {sigma_max} --sing_val_scale {svs} --sing_val_init {svi} --error_delta {err}'


ps = f'{bs} {run(1)} & {bs} {run(2)} & {bs} {run(3)} & {bs} {run(4)} & {bs} {run(5)} & {bs} {run(6)} & {bs} {run(7)} & {bs} {run(8)}'

subprocess.run(ps, shell=True)


# Random test
