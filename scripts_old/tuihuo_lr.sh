nproc_per_node=8

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8

# nohup bash scripts/tuihuo_lr.sh > qwen_tool_all_data_180k_alldata_wogemini_retool2k_mmeu10k_filter_tuihuo_lr1e_6.log 2>&1 &
bsz=1
#501760
output_dir="/mmu_mllm_hdd_2/yifanzhang/models/tool_final/qwen_tool_all_data_180k_alldata_wogemini_retool2k_mmeu10k_filter_tuihuo_lr1e_6"

FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=3211264 \
swift sft \
    --model /mmu_mllm_hdd_2/yifanzhang/models/tool_final/qwen_tool_all_data_180k_alldata_wpgemini/v0-20250616-170052/checkpoint-7740 \
    --dataset /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/retool/swift_retool_2k_filter.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_true_filtered_codeblock.jsonl  \
    --train_type full \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --system scripts/training_data/prompt.txt \
    --num_train_epochs 3 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
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

    # /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_true_chunk0_of_1.jsonl \
