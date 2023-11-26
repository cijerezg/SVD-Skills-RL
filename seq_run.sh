# Benchmarks
# python runs.py --algo Replayratio-v2 --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.001
# python runs.py --algo Replayratio-v3 --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.001
# python runs.py --algo SPiRL --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.01
# python runs.py --algo Layernorm --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.01
# python runs.py --algo Underparameter --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.01


# SERENE
# python runs.py --algo SERENE-S-4 --sing_val_scale 1 --sing_val_init 1 --sigma_max 4 --error_delta 0.001
# python runs.py --algo SERENE-S-1 --sing_val_scale 1 --sing_val_init 1 --sigma_max 1 --error_delta 0.001
python runs.py --algo SERENE-S-1-E-0.005 --sing_val_scale 1 --sing_val_init 1 --sigma_max 1 --error_delta 0.005
python runs.py --algo SERENE-S-1-E-0.025 --sing_val_scale 1 --sing_val_init 1 --sigma_max 1 --error_delta 0.025
python runs.py --algo SERENE-S-1-E-0.0002 --sing_val_scale 1 --sing_val_init 1 --sigma_max 1 --error_delta 0.0002
python runs.py --algo SERENE-S-1-E-0.00004 --sing_val_scale 1 --sing_val_init 1 --sigma_max 1 --error_delta 0.00004
# python runs.py --algo SERENE-No-S --sing_val_scale 1 --sing_val_init 1 --sigma_max .0001 --error_delta 0.001


