# This script is for reproducing the MambaSL classification results on the 30 UEA datasets.

# Download checkpoints 
#   from https://drive.google.com/drive/folders/1dJx_rpB7UnkMuxrCEoHJcXXzhaACS5Sx?usp=share_link (checkpoint_best/MambaSL.zip)
#   and change the `checkpoint_dir` variable to the path of the downloaded checkpoints.

# If you want to reproduce the other baseline results reported in MambaSL paper (https://openreview.net/pdf?id=YDl4vqQqGP),
#   please refer to the official MambaSL repo: https://github.com/yoom618/MambaSL


# Global Setting
model_name="MambaSingleLayer"
gpu_id=0
resource_dir="/data/yoom618/TSLib"
data_dir="${resource_dir}/dataset"
checkpoint_dir="${resource_dir}/checkpoints_best/MambaSL"


# ArticularyWordRecognition
dataset_name="ArticularyWordRecognition"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 128 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# AtrialFibrillation
dataset_name="AtrialFibrillation"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 13 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# BasicMotions
dataset_name="BasicMotions"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# CharacterTrajectories
dataset_name="CharacterTrajectories"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 128 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 4 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Cricket
dataset_name="Cricket"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 24 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# DuckDuckGeese
dataset_name="DuckDuckGeese"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 2 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 6 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# EigenWorms
dataset_name="EigenWorms"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 360 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 4 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Epilepsy
dataset_name="Epilepsy"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 5 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# ERing
dataset_name="ERing"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 128 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# EthanolConcentration
dataset_name="EthanolConcentration"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 512 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 36 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# FaceDetection
dataset_name="FaceDetection"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 256 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# FingerMovements
dataset_name="FingerMovements"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# HandMovementDirection
dataset_name="HandMovementDirection"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 256 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 8 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Handwriting
dataset_name="Handwriting"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 4 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Heartbeat
dataset_name="Heartbeat"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 64 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 9 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# InsectWingbeat
dataset_name="InsectWingbeat"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# JapaneseVowels
dataset_name="JapaneseVowels"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 128 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# Libras
dataset_name="Libras"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# LSST
dataset_name="LSST"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# MotorImagery
dataset_name="MotorImagery"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 60 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# NATOPS
dataset_name="NATOPS"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 512 \
  --d_ff 2 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PEMS-SF
dataset_name="PEMS-SF"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 512 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PenDigits
dataset_name="PenDigits"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 64 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# PhonemeSpectra
dataset_name="PhonemeSpectra"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 256 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 5 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# RacketSports
dataset_name="RacketSports"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 4 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SelfRegulationSCP1
dataset_name="SelfRegulationSCP1"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 256 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 18 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SelfRegulationSCP2
dataset_name="SelfRegulationSCP2"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 256 \
  --d_ff 16 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 1 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 24 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# SpokenArabicDigits
dataset_name="SpokenArabicDigits"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 8 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 1 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 3 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# StandWalkJump
dataset_name="StandWalkJump"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 32 \
  --d_ff 1 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 1 \
  --tv_B 0 \
  --tv_C 0 \
  --use_D 0 \
  --num_kernels 50 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# UWaveGestureLibrary
dataset_name="UWaveGestureLibrary"
python run.py \
  --use_gpu \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/${dataset_name}" \
  --checkpoints "${checkpoint_dir}" \
  --model "${model_name}" \
  --model_id "${dataset_name}" \
  --d_model 1024 \
  --d_ff 2 \
  --expand 1 \
  --d_conv 4 \
  --tv_dt 0 \
  --tv_B 0 \
  --tv_C 1 \
  --use_D 0 \
  --num_kernels 7 \
  --is_training 0 \
  --pred_len 0 \
  --label_len 0 \
  --batch_size 16 \
  --des gating4proposed \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10