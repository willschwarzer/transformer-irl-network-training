#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --data-prefix rings_1000 --save-to nshot_2.parameters --wandb-project sirl_rings --no-trajectory-sigmoid --trajectory-hidden-size 2048 --num-trajectory-layers 2 --state-hidden-size 2048 --num-state-layers 2 --batch-size 256 --env rings --trajectory-rep-dim 100 --state-rep-dim 100 --num-states 50 --rand-n --lr 0.001 --num-epochs 1000 --saved-model nshot_2.parameters
# wandb sync --sync-all