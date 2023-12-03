#!/bin/bash

DATSET="capella"
OUTDIR="results"
SEED=0
EPOCHS=100

# depend on the experiments
train_options="--use_pe --rotate 1"
log_options="--wandb --log_plt --log_fig"

# train
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS $train_options $log_options

# test
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS --test_mode $train_options $log_options