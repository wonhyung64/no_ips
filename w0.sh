#!/bin/bash

####srun --cpus-per-task=1 -p gpu6 --gres=gpu:a10:1 --exclude=n026 --pty bash

# BASE_DIR=/home1/wonhyung64
BASE_DIR=/home1/ok69531

REPO_DIR=/home1/ok69531
ENV=python

# REPO_DIR=/home1/wonhyung64
# ENV=$BASE_DIR/anaconda3/envs/openmmlab/bin/python3

# data directory for experiments
DATA_DIR=$REPO_DIR/no_ips/causality/data

RANDOM_SEED=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch


experiments=(
"--dataset-name=original --lr=1e-3 --weight-decay=1e-4 --alpha=0.001 --propensity=pred --omega=0.75"
)
EXECUTION_FILE=$REPO_DIR/no_ips/causality/multi_plus_ips.py
for index in ${!experiments[*]}; do
    wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--dataset-name=original --lr=1e-3 --weight-decay=1e-4 --alpha=1. --omega=0.25"
)
EXECUTION_FILE=$REPO_DIR/no_ips/causality/multi_plus_naive.py
for index in ${!experiments[*]}; do
    wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--dataset-name=original --lr=1e-2 --weight-decay=1e-4 --propensity=pred --omega=0.95"
)
EXECUTION_FILE=$REPO_DIR/no_ips/causality/ips_plus.py
for index in ${!experiments[*]}; do
    wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--dataset-name=original --lr1=1e-4 --lamb1=1e-4 --omega=9999."
)
EXECUTION_FILE=$REPO_DIR/no_ips/causality/akb_ips_plus.py
for index in ${!experiments[*]}; do
    wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--dataset-name=original --lr=1e-2 --weight-decay=1e-5 --omega=9999."
)
EXECUTION_FILE=$REPO_DIR/no_ips/causality/naive_plus.py
for index in ${!experiments[*]}; do
    wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


# experiments=(
# "--dataset-name=original --lr=1e-2 --weight-decay=1e-4 --alpha=0.01 --beta=2. --propensity=pred --omega=9999."
# )
# EXECUTION_FILE=$REPO_DIR/no_ips/causality/escm2_plus_ips.py
# for index in ${!experiments[*]}; do
#     wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
#     $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
# done


# experiments=(
# "--dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=2. --beta=2. --omega=9999."
# )
# EXECUTION_FILE=$REPO_DIR/no_ips/causality/escm2_plus_naive.py
# for index in ${!experiments[*]}; do
#     wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
#     $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
# done


# experiments=(
# "--dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=1. --omega=0.25"
# )
# EXECUTION_FILE=$REPO_DIR/no_ips/causality/esmm_plus.py
# for index in ${!experiments[*]}; do
#     wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
#     $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
# done


# experiments=(
# "--dataset-name=original --lr=1e-2 --weight-decay=1e-4 --alpha=0.001 --beta=2. --zeta=0.01 --propensity=pred --omega=9999."
# )
# EXECUTION_FILE=$REPO_DIR/no_ips/causality/v2_plus_ips.py
# for index in ${!experiments[*]}; do
#     wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
#     $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
# done


# experiments=(
# "--dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=2. --beta=2. --zeta=1. --omega=9999."
# )
# EXECUTION_FILE=$REPO_DIR/no_ips/causality/v2_plus_naive.py
# for index in ${!experiments[*]}; do
#     wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
#     $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
# done
