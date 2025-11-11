#!/bin/bash

#SBATCH --job-name=NHiTS_ETTm2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4 
#SBATCH --partition=P2
#SBATCH --output=/home/s2/jinmyeongchoi/nhits-tsl/log/S-%x.%j.out


# Loop over datasets and prediction lengths
model_name=NHiTS
dataset=ETTm2
# seq_lens=(96 192 336 720)

pred_lens=(96 192 336 720)
seq_lens=(480 960 1680 3600)

# n_pool_kernel_size from paper
k_list=(8 4 1)   # or 16 8 1, etc.

# choose r^{-1} set per H so that d = H / r^{-1} is integer
# H=96  -> r^{-1}=[24,12,1]   -> d=[4,8,96]
# H=192 -> r^{-1}=[24,12,1]   -> d=[8,16,192]
# H=336 -> r^{-1}=[168,24,1]  -> d=[2,14,336]
# H=720 -> r^{-1}=[180,60,1]  -> d=[4,12,720]

for i in "${!pred_lens[@]}"; do
  H=${pred_lens[$i]}
  L=${seq_lens[$i]}

  if   [ "$H" -eq 96 ];  then d1=4; d2=8;  d3=96
  elif [ "$H" -eq 192 ]; then d1=8; d2=16; d3=192
  elif [ "$H" -eq 336 ]; then d1=2; d2=14; d3=336
  elif [ "$H" -eq 720 ]; then d1=4; d2=12; d3=720
  else echo "No r^{-1} mapping for H=$H"; exit 1; fi

  python -u /home/s2/jinmyeongchoi/nhits-tsl/run.py \
    --is_training 1 \
    --seed 7 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --task_name long_term_forecast \
    --model_id ${model_name} \
    --model ${model_name} \
    --data ${dataset} \
    --root_path /shared/s2/lab01/timeSeries/forecasting/base/ETT-small/ \
    --data_path ETTm2.csv \
    --features M \
    --seq_len ${L} \
    --label_len 0 \
    --pred_len ${H} \
    --pooling_mode max \
    --interpolation_mode linear \
    --activation ReLU \
    --batch_normalization \
    --dropout 0.1 \
    --lradj nhits \
    --initialization orthogonal \
    --stack_types identity identity identity \
    --n_blocks 1 1 1 \
    --n_layers 2 \
    --n_theta_hidden 512 512 \
    --n_pool_kernel_size ${k_list[@]} \
    --n_freq_downsample ${d1} ${d2} ${d3}
done