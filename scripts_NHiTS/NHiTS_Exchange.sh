#!/bin/bash

#SBATCH --job-name=NHiTS_Exchange
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4 
#SBATCH --partition=P2
#SBATCH --output=/home/s2/jinmyeongchoi/nhits-tsl/log/S-%x.%j.out


# Loop over datasets and prediction lengths
model_name=NHiTS
dataset=exchange_rate
# seq_lens=(96 192 336 720)
pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
  python -u /home/s2/jinmyeongchoi/nhits-tsl/run.py \
    --is_training 1 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --task_name long_term_forecast \
    --model_id ${model_name} \
    --model ${model_name} \
    --data custom \
    --root_path /shared/s2/lab01/timeSeries/forecasting/base/exchange_rate/ \
    --data_path exchange_rate.csv \
    --features S \
    --seq_len $((pred_lens[$i]*5)) \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --pooling_mode max \
    --interpolation_mode linear \
    --activation ReLU \
    --batch_normalization \
    --dropout 0.1 \
    --initialization orthogonal \
    --stack_types identity identity identity \
    --n_blocks 1 1 1 \
    --n_layers 2 \
    --n_theta_hidden 512 512 \
    --n_pool_kernel_size 8 4 1 \
    --n_freq_downsample 16 8 1
done
