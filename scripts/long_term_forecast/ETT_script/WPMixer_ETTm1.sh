
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTm1
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.001277976 0.002415901 0.001594735 0.002011441)
batches=(256 256 256 256)
epochs=(80 80 80 80)
dropouts=(0.4 0.4 0.4 0.4)
patch_lens=(48 48 48 48)
lradjs=(type3 type3 type3 type3)
d_models=(256 128 256 128)
patiences=(12 12 12 12)

# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(db2 db3 db5 db5)
levels=(1 1 1 4)
tfactors=(5 3 7 3)
dfactors=(3 7 7 8)
strides=(24 24 24 24)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--is_training 1 \
		--root_path ./data/ETT/ \
		--data_path ETTm1.csv \
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
