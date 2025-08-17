pkill -f rlhf
# 401408
# 3211264
# rm /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/*.jsonl
# nohup bash scripts/rl/rl_1_new_reward_2.sh > rl_fmt_orm_cst05_gen4_penalty_11_lr1e_6_maxpixel_bsz128_iter2_wokl_lora.log 2>&1 &
sleep 5
# swift rlhf \
export WANDB_API_KEY='5a3fddd88cafdc3a9ed01b89e871a60b28e76d2a'
export NCCL_TIMEOUT=12000000  # 设置超时时间为1200000毫秒（10分钟）
# wandb login --host=http://11.33.252.124:8890
timestamp=$(date +%Y%m%d-%H:%M:%S)
MAX_PIXELS=3211264 \
MASTER_PORT=38325 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
REWARD_API_ADDRESS=10.82.121.22 \
QWEN_API_PORT=8000 \
VQA_WEIGHT=1 \
FMT_WEIGHT=0.5 \
CODE_WEIGHT=0.3 \
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
    --model /mmu_mllm_hdd_2/yifanzhang/models/tool_final/qwen_tool_all_data_180k_alldata_wogemini_retool2k_mmeu10k_filter_tuihuo_lr1e_6/v0-20250618-155155/checkpoint-192 \
    --external_plugins /mmu_mllm_hdd_2/yifanzhang/agent_latest_code/examples/train/grpo/plugin/agent_rm.py \
    --reward_funcs fmt_orm vqa_orm cst_orm \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_device auto \
    --vllm_tensor_parallel_size 1 \
    --vllm_limit_mm_per_prompt '{"image": 12, "video": 0}' \
    --vllm_max_model_len 12000 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset_shuffle true \
    --dataset /mllm_hdd/yfzhang/Agent-R1/construct_data/deepeyes/RL.jsonl \
    --max_length 12000 \
    --max_completion_length 3076 \
    --max_pixels 3211264 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --stop_words "</code>" "</answer>" "<|im_end|>" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-6 \
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
    --output_dir /mmu_mllm_hdd_2/yifanzhang/models/tool_final/rl_fmt_orm_cst05_gen4_penalty_11_lr1e_6_maxpixel_bsz128_iter2_wokl_lora \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.05 \
    --num_iterations 1 \
    --async_generate false \
    --deepspeed zero3 \
    --overlong_filter true \
    --log_completions true \
    --num_iterations 2 \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true \
    --report_to tensorboard \
    --attn_impl flash_attn 

    # --reward_funcs fmt_orm code_orm vqa_orm code_prm vqa_prm \

# cuda:4 cuda:5 cuda:6 cuda:7
#     --process_reward_funcs region_prm \

    # --top_p 0.9 \
    # --top_k 50 \
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