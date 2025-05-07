#!/bin/bash

####srun --cpus-per-task=1 -p gpu6 --gres=gpu:a10:1 --pty bash

BASE_DIR=/home1/wonhyung64
REPO_DIR=/home1/wonhyung64/Github
ENV=$BASE_DIR/anaconda3/envs/openmmlab/bin/python3

# data directory for experiments
DATA_DIR=$REPO_DIR/no_ips/causality/data

RANDOM_SEED=0

experiments=(
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-4 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-4 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-4 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-5 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-5 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-5 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-6 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-6 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=1. --alpha=1. --weight-decay=1e-6 --lr=1e-4"


    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-4 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-4 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-4 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-5 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-5 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-5 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-6 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-6 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=2. --alpha=1. --weight-decay=1e-6 --lr=1e-4"


    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-4 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-4 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-4 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-5 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-5 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-5 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-6 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-6 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=4. --alpha=1. --weight-decay=1e-6 --lr=1e-4"


    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-4 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-4 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-4 --lr=1e-4"
                                          #here                                
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-5 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-5 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-5 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-6 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-6 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=8. --alpha=1. --weight-decay=1e-6 --lr=1e-4"


    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-4 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-4 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-4 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-5 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-5 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-5 --lr=1e-4"
                                                                          
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-6 --lr=1e-2"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-6 --lr=1e-3"
    # "--base-model=ncf --dataset-name=original --loss-type=naive --gamma=0. --alpha=1. --weight-decay=1e-6 --lr=1e-4"

)

EXECUTION_FILE=$REPO_DIR/no_ips/causality/k_fold_cv/multi_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done
