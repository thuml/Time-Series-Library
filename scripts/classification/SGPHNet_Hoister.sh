export CUDA_VISIBLE_DEVICES=0

model_name=SGPHNet
root_path=./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --model_id H5_SGPH \
  --model $model_name \
  --data CSV_CLS \
  --root_path $root_path \
  --label_col running_state_five_class \
  --drop_cols id,time,JianSuDuan_ChaoSu,running_state_class,running_state_five_class \
  --seq_len 96 \
  --window_step 8 \
  --window_label_mode last \
  --file_split_mode shuffle \
  --split_seed 2 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --features M \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --use_class_weights \
  --cls_loss focal \
  --focal_gamma 2.0 \
  --enable_progression_targets \
  --state_graph_profile hoister_overspeed \
  --warning_horizon 5 \
  --time_bucket_steps 1,3,5,10 \
  --aux_hazard_weight 0.5 \
  --aux_time_weight 0.3 \
  --aux_next_state_weight 0.3 \
  --aux_invalid_transition_weight 0.05 \
  --itr 1 \
  --num_workers 0 \
  --des SGPH_Hoister
