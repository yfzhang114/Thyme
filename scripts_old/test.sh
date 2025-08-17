nproc_per_node=1

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
# nohup bash scripts/rm.sh > qwenvl25_7b_wociritc_mmrlhf_exist_7B_rmhead_l2_silu_newhead.log 2>&1 &
bsz=1

output_dir="./models/reward_models/training_file_swift_28k_improved_logic_refine_q"

FPS_MAX_FRAMES=10 \
MAX_PIXELS=501760 \
NPROC_PER_NODE=1 \
python swift/cli/sft.py \
    --model /mllm_hdd/yfzhang/data/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --dataset scripts/test.jsonl \
    --train_type full \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $bsz) \
    --save_strategy steps \
    --max_length 20480 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --deepspeed zero2 \
    --attn_impl flash_attn

# 学科类：MMMU、MMStar
# 图表类：AI2D
# 数学：MathVision、MathVista、OlympiadBench、LogicVista、WeMath