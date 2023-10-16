import subprocess
import argparse
import pdb

parser = argparse.ArgumentParser()

args = parser.add_argument('--algo', type=str)

args = parser.parse_args()

alg = args.algo

bs = 'python train.py'
run = lambda x: f'--run {x} --algo {alg}'


ps = f'{bs} {run(1)} & {bs} {run(2)} & {bs} {run(3)} & {bs} {run(4)} & {bs} {run(5)} & {bs} {run(6)} & {bs} {run(7)} & {bs} {run(8)}'

subprocess.run(ps, shell=True)


# Random test
