# Benchmarks
python runs.py --algo Replayratio --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.001
python runs.py --algo SPiRL --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.01
python runs.py --algo Layernorm --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.01
python runs.py --algo Underparameter --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.01


# SERENE
python runs.py --algo SERENE-S-4 --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.001
python runs.py --algo SERENE-S-1 --sing_val_scale 1 --sing_val_init 1 --sigma_max 1 --error_delta 0.001
python runs.py --algo SERENE-No-S --sing_val_scale 1 --sing_val_init 1 --sigma_max .0001 --error_delta 0.001


