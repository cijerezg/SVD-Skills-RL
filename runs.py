import subprocess
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--algo', type=str)
parser.add_argument('--sigma_max', type=float)
parser.add_argument('--reset_freq', type=int, default=8000)
parser.add_argument('--grad_steps', type=int, default=4)

args = parser.parse_args()

alg = args.algo
sigma_max = args.sigma_max
rf = args.reset_freq
gd = args.grad_steps
                

bs = 'python train.py'
run = lambda x: f'--run {x} --algo {alg} --sigma_max {sigma_max} --reset_freq {rf} --grad_steps {gd}'


ps = f'{bs} {run(1)} & {bs} {run(2)} & {bs} {run(3)} & {bs} {run(4)} & {bs} {run(5)} & {bs} {run(6)} & {bs} {run(7)} & {bs} {run(8)}'

subprocess.run(ps, shell=True)


# Random test
