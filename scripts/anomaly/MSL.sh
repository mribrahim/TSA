export CUDA_VISIBLE_DEVICES=0

batch_size=128
down_sampling_layers=3
down_sampling_window=2

for method in ChaMTeC #LMSAutoTSF iTransformer TimeMixer PatchTST 
do
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 0 \
    --root_path ./dataset/MSL \
    --model_id MSL \
    --model $method \
    --data MSL \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 32 \
    --d_ff 32 \
    --e_layers 3 \
    --enc_in 55 \
    --c_out 55 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --batch_size $batch_size \
    --anomaly_ratio 1 \
    --train_epochs 10 \
    --flag_anomaly_sliding_window_threshold
done