#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 100000 --save-to spaceinvaders.parameters --permute-types --wandb-project sirl_si --no-trajectory-sigmoid -v
# wandb sync --sync-all
