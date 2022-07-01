#!/bin/bash
# wandb offline
nice -n 15 python predict_transformer_nonlinear.py --num-examples 460000 --save-to no_sigmoid_mlp.parameters --permute-types --mlp --no-trajectory-sigmoid -v
# wandb sync --sync-all
