
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=electricity
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.00328086 0.000493286 0.002505375 0.001977516)
batches=(32 32 32 32)
epochs=(100 100 100 100)
dropouts=(0.1 0.1 0.2 0.1)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(32 32 32 32)
patiences=(12 12 12 12)

# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(sym3 coif5 sym4 db2)
levels=(2 3 1 2)
tfactors=(3 7 5 7)
dfactors=(5 5 7 8)
strides=(8 8 8 8)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--is_training 1 \
		--root_path ./data/electricity/ \
		--data_path electricity.csv \
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
