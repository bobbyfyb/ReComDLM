while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if echo "$util" | awk '$1<30 && $2<30' | grep -q . && \
     echo "$mem" | awk '$1<5000 && $2<5000' | grep -q .; then
    echo "GPUs idle, starting training..."
    uv run accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 2 \
    run_llada_sft.py \
    --train_jsonl_path /home/iiserver32/Workbench/fyb/controllable-diffusion/baselines/latent-diffusion-for-language/datasets/commongen/train.jsonl \
    --valid_jsonl_path /home/iiserver32/Workbench/fyb/controllable-diffusion/baselines/latent-diffusion-for-language/datasets/commongen/valid.jsonl \
    --test_jsonl_path /home/iiserver32/Workbench/fyb/controllable-diffusion/baselines/latent-diffusion-for-language/datasets/commongen/test.jsonl \
    --preprocessed_dir data/llada_sft_commongen \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --output_dir /data2/fyb/dllm/LLaDA-8B-Instruct/sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8
    break
  fi
  echo "GPUs busy, waiting..."
  sleep 30
done
