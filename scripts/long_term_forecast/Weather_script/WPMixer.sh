
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=weather
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.000913333 0.001379042 0.000607991 0.001470479)
batches=(32 64 32 128)
epochs=(60 60 60 60)
dropouts=(0.4 0.4 0.4 0.4)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(256 128 128 128)
patiences=(12 12 12 12)

# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(db3 db3 db3 db2)
levels=(2 1 2 1)
tfactors=(3 3 7 7)
dfactors=(7 7 7 5)
strides=(8 8 8 8)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--is_training 1 \
		--root_path ./data/weather/ \
		--data_path weather.csv \
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
