
# NCCL_DEBUG=INFO \
# REWARD_API_ADDRESS=0.0.0.0 \
# QWEN_API_PORT=8000 \
# VQA_WEIGHT=1 \
# FMT_WEIGHT=0.5 \
# CODE_WEIGHT=0.1 \
# MAX_PIXELS=3211264 \
deepspeed --hostfile=hostfile_5655_1 \
    scripts/rl/rlhf_ds.py \
    --rlhf_type grpo \
    --model your_sft_model \
    --external_plugins ./examples/train/grpo/plugin/agent_rm.py \
    --reward_funcs fmt_orm vqa_orm cst_orm \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_max_model_len 10240 \
    --vllm_tensor_parallel_size 1 \
    --vllm_limit_mm_per_prompt '{"image": 6, "video": 0}' \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.55 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset your_data \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --max_pixels 2408448 \
    --max_length 10240 \
    --max_completion_length 3196 \
    --freeze_aligner false \
    --stop_words '\<\|im_end\|\>' '\</code\>' '\</answer\>' '\<code\>' \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --padding_side left \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1, "num_cycles": 0.5}' \
    --gradient_accumulation_steps 5 \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --split_dataset_ratio 0 \
    --eval_steps 20000 \
    --save_steps 100 \
    --save_total_limit 100000 \
    --logging_steps 1 \
    --output_dir output_dir \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --num_generations 4 \
    --temperature 1.0 \
    --beta 0.01 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.05 \
    --deepspeed zero3 \
    --O3 true \
    --log_completions true \
    --report_to tensorboard \
    --async_generate false \
    --num_iterations 1 \
    --overlong_filter true \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true \
    --attn_impl flash_attn \
    --vllm_enforce_eager true \