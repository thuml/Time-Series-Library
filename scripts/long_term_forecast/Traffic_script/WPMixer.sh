
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=traffic
seq_lens=(1200 1200 1200 1200)
pred_lens=(96 192 336 720)
learning_rates=(0.0010385 0.000567053 0.001026715 0.001496217)
batches=(16 16 16 16)
epochs=(60 60 50 60)
dropouts=(0.05 0.05 0.0 0.05)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(16 32 32 32)
patiences=(12 12 12 12)

# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(db3 db3 bior3.1 db3)
levels=(1 1 1 1)
tfactors=(3 3 7 7)
dfactors=(5 5 7 3)
strides=(8 8 8 8)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--is_training 1 \
		--root_path ./data/traffic/ \
		--data_path traffic.csv \
		--model_id wpmixer \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--label_len 0 \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp
done
