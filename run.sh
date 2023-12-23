#!/bin/bash

DATSET="../oem/data/capella-oem"
DATSET="../new_research/capella-oem/capella-oem/"
OUTDIR="../oem/data/results/"
OUTDIR="results/1216/"
SEED=1
EPOCHS=30

# depend on the experiments
train_options="--rotate 1 --batch_size 4 "
log_options="--log_plt --log_fig"
wandb="--wandb"
network="--network unetformer"

# train

#unetformer_option="--unetformer_option only_angle1"
#python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS $train_options $log_options $network $unetformer_option
#unetformer_option="--unetformer_option only_angle2"
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS $train_options $log_options $network $unetformer_option --test_mode
unetformer_option="--unetformer_option with_angle1"
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS $train_options $log_options $network $unetformer_option  --test_mode
unetformer_option="--unetformer_option with_angle2"
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS $train_options $log_options $network $unetformer_option --test_mode
unetformer_option="--unetformer_option none"
python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS $train_options $log_options $network $unetformer_option  --test_mode
#test
#python3 train_unet_capella.py --dataset $DATSET --outdir $OUTDIR --seed $SEED --n_epochs $EPOCHS --test_mode $train_options $log_options $network
