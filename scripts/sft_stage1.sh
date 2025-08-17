nproc_per_node=8

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8

# nohup bash scripts/rm.sh > qwen_tool_all_data_180k_3epoch_4096_all_2round_maskstep1_code.log 2>&1 &
bsz=1
#501760
output_dir="/mmu_mllm_hdd_2/yifanzhang/models/qwen_tool_all_data_180k_3epoch_4096_all_2round_maskstep1_code"

FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=3211264 \
swift sft \
    --model /mllm_hdd/yfzhang/data/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --dataset /mmu_mllm_hdd_2/yifanzhang/agent_latest_code/scripts/training_data/wo_system_image_180k_filter_w_image_size_filter_wo_some_code.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/training_data/vstar_2step_nooverlap_training_filtered_70k.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/training_data/vstar_2step_zoomin_training_filtered_70k.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/training_data/wo_system_image_180k_filter_w_image_size_filter_wo_some_code_2round.jsonl \
    --train_type full \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --system scripts/prompt.txt \
    --num_train_epochs 3 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $bsz) \
    --save_strategy epoch \
    --max_length 10240 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --report_to wandb
