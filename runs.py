import subprocess

subprocess.run("python train.py --run 1 & python train.py --run 2 & python train.py --run 3 & python train.py --run 4",
               shell=True)


# Random test
