pkill -f rlhf
# 401408
# 3211264
# rm /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/*.jsonl
# nohup bash scripts/rl/rl_ablation/ablatin_1.sh > baseline_final_setting_from_qwen.log 2>&1 &
sleep 5
# swift rlhf \
export WANDB_API_KEY='5a3fddd88cafdc3a9ed01b89e871a60b28e76d2a'
export NCCL_TIMEOUT=12000000  # 设置超时时间为1200000毫秒（10分钟）
# wandb login --host=http://11.33.252.124:8890
timestamp=$(date +%Y%m%d-%H:%M:%S)
MAX_PIXELS=1605632 \
MASTER_PORT=38325 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
REWARD_API_ADDRESS=10.48.45.89 \
QWEN_API_PORT=8000 \
VQA_WEIGHT=1 \
GEN_ITER_LIMIT=4 \
FMT_WEIGHT=0.5 \
CODE_WEIGHT=0.1 \
VQA_PRM_WEIGHT=0.3 \
CODE_PRM_WEIGHT=0.3 \
VQA_NORM=0 \
FMT_NORM=0 \
CODE_NORM=0 \
VQA_PRM_NORM=0 \
CODE_PRM_NORM=0 \
TIME_STAMP=${timestamp} \
deepspeed \
    scripts/rl/rlhf_ds.py \
    --rlhf_type grpo \
    --external_plugins /mmu_mllm_hdd_2/yifanzhang/agent_latest_code/examples/train/grpo/plugin/agent_rm.py \
    --reward_funcs fmt_orm vqa_orm cst_orm \
    --model /mllm_hdd/yfzhang/data/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_device auto \
    --vllm_tensor_parallel_size 1 \
    --vllm_limit_mm_per_prompt '{"image": 8, "video": 0}' \
    --vllm_max_model_len 10000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset_shuffle true \
    --dataset /mllm_hdd/yfzhang/Agent-R1/construct_data/deepeyes/RL.jsonl /mmu_mllm_hdd_2/yifanzhang/high_resolution/RL.jsonl \
    --max_length 10000 \
    --max_completion_length 3196 \
    --max_pixels 1605632 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --stop_words "</code>" "</answer>" "<|im_end|>" "<code>" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1, "num_cycles": 0.5}' \
    --padding_side left \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --O3 true \
    --eval_steps 20000 \
    --save_steps 100 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --output_dir /mmu_mllm_hdd_2/yifanzhang/models/tool_ablation/baseline_fmt_orm_cst_gen4_pixel2048_kl1e_3_lr1e_6_cst05_oldcode01_iterlimit6 \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.9 \
    --beta 0.001 \
    --top_k 50 \
    --repetition_penalty 1.05 \
    --num_iterations 1 \
    --async_generate false \
    --deepspeed zero3 \
    --overlong_filter true \
    --log_completions true \
    --num_iterations 1 \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true \
    --report_to tensorboard \
    --attn_impl flash_attn