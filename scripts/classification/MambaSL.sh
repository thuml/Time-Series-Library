# This script is for reproducing the MambaSL classification results on the 30 UEA datasets.

# Download checkpoints 
#   from https://drive.google.com/drive/folders/1dJx_rpB7UnkMuxrCEoHJcXXzhaACS5Sx?usp=share_link (checkpoint_best/MambaSL.zip)
#   and change the `checkpoint_dir` variable to the path of the downloaded checkpoints.

# If you want to reproduce the other baseline results reported in MambaSL paper (https://openreview.net/pdf?id=YDl4vqQqGP),
#   please refer to the official MambaSL repo: https://github.com/yoom618/MambaSL


# Global Setting
model_name="MambaSingleLayer"
gpu_id=0
resource_dir="."
data_dir="${resource_dir}/dataset"
checkpoint_dir="${resource_dir}/checkpoints_best/MambaSL"

run_model() {
  local dn=$1; local dm=$2; local df=$3; local dt=$4; local tb=$5; local tc=$6; local nk=$7; local bs=${8:-16}

  python run.py \
    --use_gpu --gpu_type cuda --gpu ${gpu_id} \
    --task_name classification --data UEA \
    --root_path "${data_dir}/${dn}" \
    --checkpoints "${checkpoint_dir}" \
    --model "${model_name}" \
    --model_id "${dn}" \
    --d_model $dm --d_ff $df --expand 1 --d_conv 4 \
    --tv_dt $dt --tv_B $tb --tv_C $tc --use_D 0 \
    --num_kernels $nk \
    --is_training 0 --pred_len 0 --label_len 0 --batch_size $bs \
    --des gating4proposed --itr 1 --dropout 0.1 \
    --learning_rate 0.001 --train_epochs 100 --patience 10
}

# ArticularyWordRecognition
run_model "ArticularyWordRecognition" 128 8 0 0 1 3 16

# AtrialFibrillation
run_model "AtrialFibrillation" 32 16 1 0 0 13 16

# BasicMotions
run_model "BasicMotions" 32 1 0 0 0 3 16

# CharacterTrajectories
run_model "CharacterTrajectories" 128 1 1 0 0 4 16

# Cricket
run_model "Cricket" 32 4 0 1 0 24 16

# DuckDuckGeese
run_model "DuckDuckGeese" 1024 2 0 0 1 6 16

# EigenWorms
run_model "EigenWorms" 32 1 1 1 0 360 4

# Epilepsy
run_model "Epilepsy" 32 1 1 1 0 5 16

# ERing
run_model "ERing" 128 8 1 0 1 3 16

# EthanolConcentration
run_model "EthanolConcentration" 512 4 0 0 0 36 16

# FaceDetection
run_model "FaceDetection" 256 16 1 0 1 3 16

# FingerMovements
run_model "FingerMovements" 32 1 0 1 1 3 16

# HandMovementDirection
run_model "HandMovementDirection" 256 16 1 0 1 8 16

# Handwriting
run_model "Handwriting" 1024 4 1 0 1 4 16

# Heartbeat
run_model "Heartbeat" 64 16 0 0 0 9 16

# InsectWingbeat
run_model "InsectWingbeat" 1024 8 0 0 0 3 16

# JapaneseVowels
run_model "JapaneseVowels" 128 8 1 1 0 3 16

# Libras
run_model "Libras" 1024 4 1 1 1 3 16

# LSST
run_model "LSST" 1024 4 1 1 1 3 16

# MotorImagery
run_model "MotorImagery" 32 8 0 0 0 60 16

# NATOPS
run_model "NATOPS" 512 2 0 1 0 3 16

# PEMS-SF
run_model "PEMS-SF" 512 1 1 1 0 3 16

# PenDigits
run_model "PenDigits" 64 1 0 1 1 3 16

# PhonemeSpectra
run_model "PhonemeSpectra" 256 4 1 1 0 5 16

# RacketSports
run_model "RacketSports" 1024 4 1 0 1 3 16

# SelfRegulationSCP1
run_model "SelfRegulationSCP1" 256 16 1 0 1 18 16

# SelfRegulationSCP2
run_model "SelfRegulationSCP2" 256 16 1 1 1 24 16

# SpokenArabicDigits
run_model "SpokenArabicDigits" 1024 8 0 1 0 3 16

# StandWalkJump
run_model "StandWalkJump" 32 1 1 0 0 50 16

# UWaveGestureLibrary
run_model "UWaveGestureLibrary" 1024 2 0 0 1 7 16