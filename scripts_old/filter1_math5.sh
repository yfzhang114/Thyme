nproc_per_node=8

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8

# nohup bash scripts/filter1_math5.sh > qwen_tool_all_data_200k_w_imagesize_3epoch_4096_all_2round_code_4ofilter1_mathrep5.log 2>&1 &
bsz=1
#501760
output_dir="/mmu_mllm_hdd_2/yifanzhang/models/qwen_tool_all_data_200k_w_imagesize_3epoch_4096_all_2round_code_4ofilter1_mathrep5"

FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=8 \
MAX_PIXELS=3211264 \
swift sft \
    --model /mllm_hdd/yfzhang/data/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --dataset /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/training_data/wo_system_image_180k_filter_w_image_size_filter_wo_some_code_4o_filter_1.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/training_data/vstar_2step_nooverlap_training_filtered_70k.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/training_data/vstar_2step_zoomin_training_filtered_70k.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/training_data/wo_system_image_180k_filter_w_image_size_filter_wo_some_code_2round.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_chunk0_of_1.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_chunk0_of_1.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_chunk0_of_1.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_chunk0_of_1.jsonl \
    /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed_swift_train_chunk0_of_1.jsonl \
    --train_type full \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --system scripts/training_data/prompt.txt \
    --num_train_epochs 3 \
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
    --dataloader_num_workers 0 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --report_to none

# /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/mm-eumath/data_gemini_code_processed.jsonl 15k
# construct_data/math_and_chart/think-litevl/data_70k_gemini_training_filtered.jsonl 40k
    # --dataset /mllm_hdd/yfzhang/Agent-R1/huggface_data/training_file_swift_28k_improved_logic_refine_q_a_final_2_filter.jsonl \
    # /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/arxivqa/arxivqa_processed_excuted_2_quality_training_v2_28k_v5.jsonl \
    # /mllm_hdd/yfzhang/Agent-R1/construct_data/V*/merged_process_box_data_38k_roi_step2_training_swift_filter.jsonl \
    # /mllm_hdd/yfzhang/Agent-R1/construct_data/training_file/training_file_wo_image_process_swift_100k.jsonl \
    # /mllm_hdd/yfzhang/Agent-R1/construct_data/math_and_chart/retool/swift_retool_2k_filter_processed.jsonl \