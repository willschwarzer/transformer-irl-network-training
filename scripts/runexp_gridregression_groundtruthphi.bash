#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 1010000 --save-to ground_truth_phi_test.parameters --wandb-project sirl --no-trajectory-sigmoid --no-space-invaders --trajectory-hidden-size 2048 --num-trajectory-layers 2 --state-hidden-size 2048 --num-state-layers 2 --batch-size 256 --no-repeat-trajectory-calculations -v --use-shuffled
# wandb sync --sync-all
