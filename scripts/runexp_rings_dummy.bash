#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --data-prefix rings_1000 --save-to ground_truth_phi_test.parameters --wandb-project sirl_rings --no-trajectory-sigmoid --trajectory-hidden-size 2048 --num-trajectory-layers 2 --state-hidden-size 2048 --num-state-layers 2 --batch-size 256 --no-repeat-trajectory-calculations -v --env rings --trajectory-rep-dim 100 --state-rep-dim 100
# wandb sync --sync-all