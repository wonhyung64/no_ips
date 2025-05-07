#!/bin/bash

####srun --cpus-per-task=1 -p gpu6 --gres=gpu:a10:1 --pty bash

BASE_DIR=/home1/wonhyung64
REPO_DIR=/home1/wonhyung64/Github
ENV=$BASE_DIR/anaconda3/envs/openmmlab/bin/python3

# data directory for experiments
DATA_DIR=$REPO_DIR/no_ips/causality/data

RANDOM_SEED=0

experiments=(

    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=2."
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-6 --lr=1e-3 --alpha=2."
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-6 --lr=1e-3 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-6 --lr=1e-3 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-6 --lr=1e-3 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=2."
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=2."
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.001"

)

EXECUTION_FILE=$REPO_DIR/no_ips/causality/k_fold_cv/multi_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done

experiments=(

    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-6 --lr=1e-4 --alpha=2."
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-6 --lr=1e-4 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-6 --lr=1e-4 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-6 --lr=1e-4 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-3 --alpha=2."
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-3 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-3 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-3 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=2."
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-4 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-5 --lr=1e-3 --alpha=2."
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-5 --lr=1e-3 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-5 --lr=1e-3 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-5 --lr=1e-3 --alpha=0.001"

)

EXECUTION_FILE=$REPO_DIR/no_ips/causality/k_fold_cv/escm2_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(

    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=2."
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=original --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=2."
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=naive --weight-decay=1e-5 --lr=1e-4 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=2."
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=original --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=2."
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=personalized --loss-type=ips --weight-decay=1e-4 --lr=1e-3 --alpha=0.001"

)

EXECUTION_FILE=$REPO_DIR/no_ips/causality/k_fold_cv/v2_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(

    # "--base-model=linearcf --dataset-name=original --weight-decay=1e-5 --lr=1e-3 --alpha=2."
    # "--base-model=linearcf --dataset-name=original --weight-decay=1e-5 --lr=1e-3 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=original --weight-decay=1e-5 --lr=1e-3 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=original --weight-decay=1e-5 --lr=1e-3 --alpha=0.001"

    # "--base-model=linearcf --dataset-name=personalized --weight-decay=1e-5 --lr=1e-3 --alpha=2."
    # "--base-model=linearcf --dataset-name=personalized --weight-decay=1e-5 --lr=1e-3 --alpha=0.1"
    # "--base-model=linearcf --dataset-name=personalized --weight-decay=1e-5 --lr=1e-3 --alpha=0.01"
    # "--base-model=linearcf --dataset-name=personalized --weight-decay=1e-5 --lr=1e-3 --alpha=0.001"

)

EXECUTION_FILE=$REPO_DIR/no_ips/causality/k_fold_cv/esmm_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done
