#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("vrnn")
budgets=(2 3 4 5 6 7)
policies=("dense" "uniform_subsampling" "random")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
        for policy in "${policies[@]}"; do

            if [ "$policy" == "dense" ]; then
                python train_classifier.py \
                    --logging_prefix smartwatch_gesture_lstm32_${policy} \
                    --sampling_policy ${policy} \
                    --input_dim 4 \
                    --architecture "$architecture" \
                    --dataset gesture \
                    --seed "$seed" \
                    --subjects 1 2 3 4 5 6 7 8 \
                    --sensors acc \
                    --body_parts right_wrist \
                    --activities 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
                    --val_frac 0.1 \
                    --window_size 0 \
                    --overlap_frac 0.5 \
                    --batch_size 8 \
                    --lr 0.0005 \
                    --epochs 300 \
                    --ese 30 \
                    --log_freq 40
            else
                for budget in "${budgets[@]}"; do
                    python train_classifier.py \
                        --logging_prefix smartwatch_gesture_lstm32_budget${budget}_${policy} \
                        --budget ${budget} \
                        --sampling_policy ${policy} \
                        --input_dim 4 \
                        --architecture "$architecture" \
                        --dataset gesture \
                        --seed "$seed" \
                        --subjects 1 2 3 4 5 6 7 8 \
                        --sensors acc \
                        --body_parts right_wrist \
                        --activities 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
                        --val_frac 0.1 \
                        --window_size 0 \
                        --overlap_frac 0.5 \
                        --batch_size 8 \
                        --lr 0.0005 \
                        --epochs 300 \
                        --ese 30 \
                        --log_freq 40
                done
            fi

        done
    done
done
