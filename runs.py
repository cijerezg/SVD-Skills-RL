import subprocess
import argparse

parser = argparse.ArgumentParser()

args = parser.add_argument('--algo', type=str)

args = parser.parse_args()

alg = args.algo


ps = f'python train.py --run 1 --algo {alg} & python train.py --run 2 --algo {alg} & python train.py --run 3 --algo {alg} & python train.py --run 4 --algo {alg}'


subprocess.run(ps, shell=True)



# Random test
