#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --data-prefix rings_1000 --save-to set_transformer.parameters --wandb-project set_transformer_rings --state-hidden-size 2048 --num-state-layers 2 --batch-size 1 --env rings --demonstration-rep-dim 10 --state-rep-dim 10 --num-states 50 --rand-n --lr 0.001 --num-epochs 1000 --dem-encoder-type set_transformer --ground-truth-weights --verbose
# wandb sync --sync-all