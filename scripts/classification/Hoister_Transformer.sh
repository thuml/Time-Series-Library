export CUDA_VISIBLE_DEVICES=0

model_name=Transformer
root_path=./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579
output_root=./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579/experiment_outputs

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --model_id H5 \
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
  --e_layers 2 \
  --d_model 64 \
  --d_ff 128 \
  --dropout 0.1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --use_class_weights \
  --cls_loss focal \
  --focal_gamma 2.0 \
  --checkpoints $output_root/checkpoints \
  --results_root $output_root/results \
  --itr 1 \
  --num_workers 0 \
  --des Hoister_Transformer
