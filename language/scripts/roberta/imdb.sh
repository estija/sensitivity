#!/bin/bash

source ~/.bashrc
conda activate /home/tianyizhou/anaconda3/envs/math


python main.py --model_name 'roberta-scratch'\
    --batch_size 64 \
    --dropout 0.1 \
    --embedding_dim 320 \
    --epochs 200 \
    --gamma 0.9 \
    --hidden_dim 320 \
    --lr 1e-04 \
    --max_length 32 \
    --num_head 8 \
    --num_labels 2 \
    --num_layers 4 \
    --replace_size 3 \
    --report_num_points 5000 \
    --train_num_points 25000 \
    --valid_num_points 9795 \
    --dataset 'imdb' \
    --cudaname 'cuda:6' \
