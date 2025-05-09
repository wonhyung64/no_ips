#!/bin/bash

####srun --cpus-per-task=1 -p gpu6 --gres=gpu:a10:1 --exclude=n026 --pty bash

BASE_DIR=/home1/wonhyung64
ENV=$BASE_DIR/anaconda3/envs/openmmlab/bin/python3

# data directory for experiments
DATA_DIR=$BASE_DIR/Github/no_ips/causality/data

RANDOM_SEED=0


experiments=(
# "--dataset-name=original --lr=1e-3 --weight-decay=1e-4 --alpha=0.001 --propensity=pred"
# "--dataset-name=personalized --lr=1e-2 --weight-decay=1e-4 --alpha=0.01 --propensity=pred"

"--base-model=linearcf --dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=0.001 --propensity=pred"
"--base-model=linearcf --dataset-name=personalized --lr=1e-4 --weight-decay=1e-4 --alpha=0.001 --propensity=pred"
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/multi_plus_ips.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-3 --weight-decay=1e-4 --alpha=1."
# "--dataset-name=personalized --lr=1e-4 --weight-decay=1e-4 --alpha=1."

"--base-model=linearcf --dataset-name=original --lr=1e-4 --weight-decay=1e-5 --alpha=2."
"--base-model=linearcf --dataset-name=personalized --lr=1e-3 --weight-decay=1e-6 --alpha=2."
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/multi_plus_naive.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-2 --weight-decay=1e-4 --propensity=pred"
# "--dataset-name=personalized --lr=1e-2 --weight-decay=1e-4 --propensity=pred"

"--base-model=linearcf --dataset-name=original --lr=1e-6 --weight-decay=1e-2 --propensity=pred"
"--base-model=linearcf --dataset-name=personalized --lr=1e-6 --weight-decay=1e-2 --propensity=pred"
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/ips_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr1=1e-4 --lamb1=1e-4"
# "--dataset-name=personalized --lr1=1e-2 --lamb1=1e-4"

"--base-model=linearcf --dataset-name=original --lr1=1e- --lamb1=1e-"
"--base-model=linearcf --dataset-name=personalized --lr1=1e- --lamb1=1e-"
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/akb_ips_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-2 --weight-decay=1e-5"
# "--dataset-name=personalized --lr=1e-2 --weight-decay=1e-5"

"--base-model=linearcf --dataset-name=original --lr=1e-3 --weight-decay=1e-5"
"--base-model=linearcf --dataset-name=personalized --lr=1e-3 --weight-decay=1e-5"
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/naive_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-2 --weight-decay=1e-4 --alpha=0.01 --beta=2. --propensity=pred"
# "--dataset-name=personalized --lr=1e-2 --weight-decay=1e-4 --alpha=1. --beta=2. --propensity=pred"

"--base-model=linearcf --dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=1. --beta=2. --propensity=pred"
"--base-model=linearcf --dataset-name=personalized --lr=1e-3 --weight-decay=1e-5 --alpha=0.1 --beta=1. --propensity=pred"
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/escm2_plus_ips.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=2. --beta=2."
# "--dataset-name=personalized --lr=1e-3 --weight-decay=1e-4 --alpha=1. --beta=2."

"--base-model=linearcf --dataset-name=original --lr=1e-4 --weight-decay=1e-6 --alpha=2. --beta=1."
"--base-model=linearcf --dataset-name=personalized --lr=1e-3 --weight-decay=1e-5 --alpha=1. --beta=1."
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/escm2_plus_naive.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=1."
# "--dataset-name=personalized --lr=1e-2 --weight-decay=1e-5 --alpha=2."

"--base-model=linearcf --dataset-name=original --lr=1e-3 --weight-decay=1e-5 --alpha=0.1"
"--base-model=linearcf --dataset-name=personalized --lr=1e-3 --weight-decay=1e-5 --alpha=1."
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/esmm_plus.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-2 --weight-decay=1e-4 --alpha=0.001 --beta=2. --zeta=0.01 --propensity=pred"
# "--dataset-name=personalized --lr=1e-2 --weight-decay=1e-4 --alpha=1. --beta=2. --zeta=1. --propensity=pred"

"--base-model=linearcf --dataset-name=original --lr=1e-3 --weight-decay=1e-4 --alpha=0.001 --beta=1. --zeta=0.001 --propensity=pred"
"--base-model=linearcf --dataset-name=personalized --lr=1e-3 --weight-decay=1e-4 --alpha=0.001 --beta=0.1 --zeta=2. --propensity=pred"
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/v2_plus_ips.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
# "--dataset-name=original --lr=1e-4 --weight-decay=1e-4 --alpha=2. --beta=2. --zeta=1."
# "--dataset-name=personalized --lr=1e-2 --weight-decay=1e-5 --alpha=1. --beta=2. --zeta=1."

"--base-model=linearcf --dataset-name=original --lr=1e-4 --weight-decay=1e-5 --alpha=2. --beta=1. --zeta=0.01"
"--base-model=linearcf --dataset-name=personalized --lr=1e-4 --weight-decay=1e-5 --alpha=2. --beta=2. --zeta=2."
)
EXECUTION_FILE=$BASE_DIR/Github/no_ips/causality/v2_plus_naive.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done

