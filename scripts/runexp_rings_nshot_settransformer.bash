#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --data-prefix rings_1000 --save-to set_transformer.parameters --wandb-project set_transformer_rings --state-hidden-size 2048 --num-state-layers 2 --batch-size 64 --env rings --trajectory-rep-dim 100 --state-rep-dim 100 --num-states 50 --rand-n --lr 0.001 --num-epochs 1000 --traj-encoder-type set_transformer
# wandb sync --sync-all