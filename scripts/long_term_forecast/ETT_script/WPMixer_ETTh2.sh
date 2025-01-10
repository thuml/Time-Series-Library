
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTh2
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.000466278 0.000294929 0.000617476 0.000810205)
batches=(256 256 256 256)
epochs=(30 30 30 30)
dropouts=(0.0 0.0 0.1 0.4)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 128 128)
patiences=(12 12 12 12)

# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(db2 db2 db2 db2)
levels=(2 3 5 5)
tfactors=(5 3 5 5)
dfactors=(5 8 3 5)
strides=(8 8 8 8)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--is_training 1 \
		--root_path ./data/ETT/ \
		--data_path ETTh2.csv \
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
