nproc_per_node=8

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8

# nohup bash scripts/rm_math.sh > qwen_think_math_mmeu_code.log 2>&1 &
bsz=1

output_dir="./models/qwen_think_lite_vl_sft"

FPS_MAX_FRAMES=10 \
MAX_PIXELS=3211264 \
NPROC_PER_NODE=8 \
swift sft \
    --model /mllm_hdd/yfzhang/data/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --dataset /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_chunk0_of_1.jsonl   \
    --train_type full \
    --system scripts/training_data/prompt.txt \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $bsz) \
    --save_strategy epoch \
    --max_length 20480 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    # --report_to wandb
