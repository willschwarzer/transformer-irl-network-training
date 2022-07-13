#!/bin/bash
# wandb offline
nice -n 15 python train_agents_will.py -ne 10000
# wandb sync --sync-all
