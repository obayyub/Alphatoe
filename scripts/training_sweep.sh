#!/bin/bash

python_script="run_training.py"

# n_head, d_head, d_model, d_mlp
hyper_params=(
    #"8 16 128 256"
    #"4 8 32 128"
    #"2 4 8 32 40000"
    "2 8 16 64 10000"
    #"2 16 32 128 20000"
)

experiment_name="hyperparam sweep"
game_type="prob all"
#num_epochs=20000
batch_size=4096

for hyper_param in "${hyper_params[@]}"; do
    read -r n_head d_head d_model d_mlp num_epochs <<< "$hyper_param"
    python "$python_script" "$experiment_name" "$game_type" \
        --batch_size "$batch_size" \
        --n_heads "$n_head" \
        --d_head "$d_head" \
        --d_model "$d_model" \
        --d_mlp "$d_mlp" \
        --n_epochs "$num_epochs" \
        --save_losses \
        --eval_model
done