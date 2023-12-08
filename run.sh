#!/bin/bash

DATSET="../oem/data/capella-oem"
OUTDIR="../oem/data/results/"
SEED=0
EPOCHS=100

# depend on the experiments
train_options="--rotate 1"
log_options="--log_plt --log_fig"
wandb="--wandb"
network="--network unetformer"

# train
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS $train_options $log_options $wandb $network

# test
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS --test_mode $train_options $log_options $network