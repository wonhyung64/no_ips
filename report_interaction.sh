#!/bin/bash

####srun --cpus-per-task=1 -p gpu6 --gres=gpu:a10:1 --pty bash

BASE_DIR=/home1/wonhyung64
ENV=$BASE_DIR/anaconda3/envs/openmmlab/bin/python3

# data directory for experiments
DATA_DIR=$BASE_DIR/Github/no_ips/interaction/data

RANDOM_SEED=0

experiments=(
"--base-model=mf --dataset-name=coat --batch-size=4096 --lr1=1e-2 --lamb1=1e-4"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr1=1e-2 --lamb1=1e-5"

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr1=1e-3 --lamb1=1e-5"
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr1=1e-4 --lamb1=1e-4"
)

EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/akb_ips.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5 --alpha=0.001 --beta=0.1"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-3 --weight-decay=1e-5 --alpha=0.01 --beta=0.01"

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-4 --alpha=2. --beta=1."
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-4 --alpha=2. --beta=1."
)

EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/escm2_ips.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5 --alpha=0.001 --beta=0.1"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-2 --weight-decay=1e-6 --alpha=0.1 --beta=1."

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-4 --alpha=2. --beta=1."
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-4 --alpha=2. --beta=1."
)

EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/escm2_naive.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5 --alpha=0.01"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-2 --weight-decay=1e-6 --alpha=1."

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-4 --alpha=2."
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-5 --alpha=2."
)

EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/multi_naive.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(

"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5 --alpha=0.1"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-3 --weight-decay=1e-5 --alpha=0.01"

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5 --alpha=2."
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-6 --alpha=1."
)

EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/multi_ips.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done

experiments=(
"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5 --alpha=0.001"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-2 --weight-decay=1e-6 --alpha=1."

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-4 --weight-decay=1e-4 --alpha=0.1"
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-2 --weight-decay=1e-6 --alpha=2."
)


EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/esmm.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-6"

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-4 --weight-decay=1e-4"
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-5"
)


EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/naive.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(

"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-5"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-5"

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-4 --weight-decay=1e-4"
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-2 --weight-decay=1e-4"
)


EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/ips.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(

"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-2 --weight-decay=1e-4"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-3 --weight-decay=1e-6"

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-4 --weight-decay=1e-4"
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-6"
)


EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/naive_v2.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done


experiments=(
"--base-model=mf --dataset-name=coat --batch-size=4096 --lr=1e-3 --weight-decay=1e-4"
"--base-model=mf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-2 --weight-decay=1e-6"

# "--base-model=ncf --dataset-name=coat --batch-size=4096 --lr=1e-4 --weight-decay=1e-4"
# "--base-model=ncf --dataset-name=yahoo_r3 --batch-size=8192 --lr=1e-4 --weight-decay=1e-4"
)


EXECUTION_FILE=$BASE_DIR/Github/no_ips/interaction/ips_v2.py
for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} --data-dir=$DATA_DIR --random-seed=$RANDOM_SEED
done
