#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("vrnn")
budgets=(2 3 4 5 6 7)

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
        for budget in "${budgets[@]}"; do
          python train_policy.py \
                --single_sensor_checkpoint_prefix motion_gesture_lstm32_budget${budget}_random \
                --logging_prefix learned_policy_budget${budget} \
                --architecture "$architecture" \
                --dataset gesture_impair \
                --input_dim 4 \
                --seed "$seed" \
                --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
                --sensors acc \
                --body_parts right_wrist \
                --activities 0 1 2 3 4 5 \
                --val_frac 0.2 \
                --window_size 0 \
                --sampling_policy learned \
                --budget ${budget} \
                --policy_batch_size 8 \
                --policy_lr 0.01 \
                --policy_epochs 5 \
                --classifier_batch_size 8 \
                --classifier_lr 0.01 \
                --classifier_epochs 5 \
                --ese 30 \
                --log_freq 10
        done
    done
done

