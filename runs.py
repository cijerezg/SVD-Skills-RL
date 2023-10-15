import subprocess
import argparse

parser = argparse.ArgumentParser()

args = parser.add_argument('--algo', type=str)

args = parser.parse_args()

alg = args.algo


ps = f'python train.py --run 5 --algo {alg} & python train.py --run 6 --algo {alg} & python train.py --run 7 --algo {alg} & python train.py --run 8 --algo {alg}'


subprocess.run(ps, shell=True)



# Random test
