import subprocess
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--algo', type=str)
parser.add_argument('--sigma_max', type=float)

args = parser.parse_args()

alg = args.algo
sigma_max = args.sigma_max

bs = 'python train.py'
run = lambda x: f'--run {x} --algo {alg} --sigma_max {sigma_max}'


ps = f'{bs} {run(1)} & {bs} {run(2)} & {bs} {run(3)} & {bs} {run(4)} & {bs} {run(5)} & {bs} {run(6)} & {bs} {run(7)} & {bs} {run(8)}'

subprocess.run(ps, shell=True)


# Random test
