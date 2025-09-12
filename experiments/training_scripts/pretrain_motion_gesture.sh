#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("vrnn")
budgets=(2 3 4 5 6 7)
policies=("random" "dense" "uniform_subsampling")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
        for policy in "${policies[@]}"; do

            if [ "$policy" == "dense" ]; then
                python train_classifier.py \
                    --logging_prefix motion_gesture_lstm32_${policy} \
                    --sampling_policy ${policy} \
                    --input_dim 4 \
                    --architecture "$architecture" \
                    --dataset gesture_impair \
                    --seed "$seed" \
                    --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
                    --sensors acc \
                    --body_parts right_wrist \
                    --activities 0 1 2 3 4 5 \
                    --val_frac 0.2 \
                    --window_size 0 \
                    --overlap_frac 0.5 \
                    --batch_size 16 \
                    --lr 0.001 \
                    --epochs 150 \
                    --ese 20 \
                    --log_freq 40
            else
                for budget in "${budgets[@]}"; do
                    python train_classifier.py \
                        --logging_prefix motion_gesture_lstm32_budget${budget}_${policy} \
                        --budget ${budget} \
                        --sampling_policy ${policy} \
                        --input_dim 4 \
                        --architecture "$architecture" \
                        --dataset gesture_impair \
                        --seed "$seed" \
                        --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
                        --sensors acc \
                        --body_parts right_wrist \
                        --activities 0 1 2 3 4 5 \
                        --val_frac 0.2 \
                        --window_size 0 \
                        --overlap_frac 0.5 \
                        --batch_size 8 \
                        --lr 0.001 \
                        --epochs 150 \
                        --ese 20 \
                        --log_freq 40
                done
            fi

        done
    done
done
