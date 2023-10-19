import subprocess
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--algo', type=str)
parser.add_argument('--reset_freq', type=int)

args = parser.parse_args()

alg = args.algo
freq = args.reset_freq

bs = 'python train.py'
run = lambda x: f'--run {x} --algo {alg} --reset_freq {freq}'


ps = f'{bs} {run(1)} & {bs} {run(2)} & {bs} {run(3)} & {bs} {run(4)} & {bs} {run(5)} & {bs} {run(6)} & {bs} {run(7)} & {bs} {run(8)}'

subprocess.run(ps, shell=True)


# Random test
