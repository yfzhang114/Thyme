pkill -f rlhf
# 401408
# 3211264
# rm /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/*.jsonl
sleep 5
# swift rlhf \
# nohup bash scripts/rl/rl_3.sh > rl_max3072_repeat_zhiweikong_gene4_kl0.log 2>&1 &
# 第一版iteration 4，应该到8，不然math可能没用
bsz=1
export WANDB_API_KEY='5a3fddd88cafdc3a9ed01b89e871a60b28e76d2a'
export NCCL_TIMEOUT=1200000  # 设置超时时间为1200000毫秒（10分钟）
# wandb login --host=http://11.33.252.124:8890
timestamp=$(date +%Y%m%d-%H:%M:%S)
MAX_PIXELS=3211264 \
MASTER_PORT=38325 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
REWARD_API_ADDRESS=10.48.48.40 \
QWEN_API_PORT=8000 \
VQA_WEIGHT=1 \
deepspeed \
    scripts/rl/rlhf_ds.py \
    --rlhf_type grpo \
    --model /mmu_mllm_hdd_2/yifanzhang/models/tool_final/qwen_tool_all_data_180k_alldata_wogemini_retool2k_mmeu10k_filter_tuihuo_lr1e_6/v0-20250618-155155/checkpoint-192 \
    --external_plugins /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/examples/train/grpo/plugin/agent_rm_yf.py \
    --reward_funcs vqa_orm \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.55 \
    --vllm_device auto \
    --vllm_tensor_parallel_size 1 \
    --vllm_limit_mm_per_prompt '{"image": 8, "video": 0}' \
    --vllm_max_model_len 10240 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mllm_hdd/yfzhang/Agent-R1/construct_data/deepeyes/RL.jsonl \
    --max_length 8192 \
    --max_completion_length 3072 \
    --max_pixels 3211264 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --stop_words "</code>" "</answer>" "<|im_end|>" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1, "num_cycles": 0.5}' \
    --padding_side left \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --beta 0. \
    --O3 true \
    --eval_steps 20000 \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/models/RL_ckpt_gen4 \
    --warmup_ratio 0.0005 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate false \
    --deepspeed zero3 \
    --overlong_filter true \
    --log_completions true \
    --num_iterations 1 \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true \
    --report_to wandb \
    --attn_impl flash_attn 
# cuda:4 cuda:5 cuda:6 cuda:7
#     --process_reward_funcs region_prm \

    # --add_version false \
    # --vlm_cot true \
    # --use_liger_kernel true \
    # --offload_model \
    # --offload_optimizer \
    # --gc_collect_after_offload \
# WANDB_API_KEY="5a3fddd88cafdc3a9ed01b89e871a60b28e76d2a" \
# FPS_MAX_FRAMES=10 \
# MAX_PIXELS=501760 \
# MASTER_ADDR=${ARNOLD_WORKER_0_HOST} MASTER_PORT=${ARNOLD_WORKER_0_PORT} NODE_RANK=${ARNOLD_ID} \
# no_proxy="" NNODES=${ARNOLD_WORKER_NUM} NPROC_PER_NODE=${ARNOLD_WORKER_GPU} \
# swift rlhf \
#     --rlhf_type rm \
#     --model ./yf_model/Qwen2.5-VL-7B-Instruct \
#     --dataset data/text_reward_data/combined_chat-5k_unltra41k_wildchat17k_reward_data.jsonl data/text_reward_data/data-olmo-2-0425-1b-preference-mix-378k.jsonl data/text_reward_data/data-tulu-3-IF-augmented-on-policy-72b-65k.jsonl data/text_reward_data data/text_reward_data/ultra_headest_63k.jsonl \
#     --train_type full \
#     --torch_dtype bfloat16 \
#     --num_train_epochs 1 \
#     --temperature 0 \
#     --center_rewards_coefficient 0.0 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --freeze_vit true \
#     --gradient_accumulation_steps $(expr 128 / $ARNOLD_WORKER_GPU / $ARNOLD_WORKER_NUM) \
#     --save_total_limit 3 \
#     --logging_steps 5 \
#     --max_length 4096 \
#     --output_dir $output_dir \
#     --save_strategy steps \
#     --save_steps 3000 \
#     --warmup_ratio 0.05 \
#     --dataloader_num_workers 32 \
#     --deepspeed zero2 \
#     --attn_impl flash_attn \
#     --report_to wandb