#!/bin/bash

####srun --cpus-per-task=1 -p gpu6 --gres=gpu:a10:1 --pty bash

BASE_DIR=/home1/wonhyung64

REPO_DIR=/home1/ok69531
ENV=python

# REPO_DIR=/home1/wonhyung64/Github
# ENV=$BASE_DIR/anaconda3/envs/openmmlab/bin/python3

# data directory for experiments
DATA_DIR=$REPO_DIR/Github/no_ips/causality/data

RANDOM_SEED=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch


experiments=(

#     "--dataset-name=original --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0."
#     "--dataset-name=personalized --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0."

#     "--dataset-name=original --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.0001"
#     "--dataset-name=personalized --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.0001"

    "--dataset-name=original --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.001"
    "--dataset-name=personalized --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.001"

    "--dataset-name=original --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.01"
    "--dataset-name=personalized --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.01"

#     "--dataset-name=original --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.1"
#     "--dataset-name=personalized --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=0.1"

#     "--dataset-name=original --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=1."
#     "--dataset-name=personalized --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=1."

#     "--dataset-name=original --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=10."
#     "--dataset-name=personalized --loss-type=naive --weight-decay=1e-4 --lr=1e-4 --alpha=10."

#     "--dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0."
#     "--dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0."

#     "--dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.0001"
#     "--dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.0001"

#     "--dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.001"
#     "--dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.001"

#     "--dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.01"
#     "--dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.01"

#     "--dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.1"
#     "--dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.1"

#     "--dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=1."
#     "--dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=1."

#     "--dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=10."
#     "--dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=10."

)

EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/sensitivity.py
for index in ${!experiments[*]}; do
    wandb login a1f59c7a0e53eed9b11d25edae53fdbe676fb53a
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done

