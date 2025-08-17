# 获取所有与 train.sh 相关的进程 PID，排除当前脚本进程的 PID
PIDS=$(pgrep -f "train_multi_ds.sh|train_mpirun.sh|train.sh|train_resume.sh" | grep -v "^$$$")
# 打印所有 PIDs
echo "PIDs to monitor: ${PIDS}"
while true; do
    pwd
    ALL_FINISHED=true
    for PID in $PIDS; do
        if [ -d "/proc/$PID" ]; then
            # 读取进程状态
            STATE=$(grep "^State:" /proc/$PID/status 2>/dev/null | awk '{print $2, $3}')
            if [[ "$STATE" == "Z (zombie)" ]]; then
                echo "Process $PID is a zombie process."
            else
                echo "Process $PID is still running (State: $STATE)."
                ALL_FINISHED=false
            fi
        else
            echo "Process $PID has terminated."
        fi
    done
    if $ALL_FINISHED; then
        echo "All processes have finished."
        break
    fi
    sleep 10
done
echo "succeed"
sleep 10

# 检查是否已安装 pdsh
if ! command -v pdsh >/dev/null 2>&1; then
    echo "pdsh 未安装，开始安装..."

    # 设置代理环境变量
    export http_proxy="http://oversea-squid1.jp.txyun:11080"
    export https_proxy="http://oversea-squid1.jp.txyun:11080"
    export no_proxy="localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"

    # 更新 apt 索引并安装 pdsh
    apt update && apt install -y pdsh

    # 修改权限（如有需要）

    echo "pdsh 安装完成。"
else
    echo "pdsh 已安装。"
fi
chown root:root /usr/lib
source /share/wenbin/miniconda3/bin/activate
conda activate o3
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=11.33.252.124,localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# export HF_HOME=/hetu_group/jky/playground/2024/yuanqi_v2.0/dataset/huggingface_cache
# export MODELSCOPE_CACHE=/hetu_group/jky/playground/2024/yuanqi_v2.0/dataset/modelscope_cache
export PATH=/root/pdsh-2.29/bin:$PATH

# build log dir
timestamp=$(date +%m%d_%H:%M:%S)
echo "Now time: $timestamp" 

# 如果 output 不存在就创建
if [ ! -d "./output/log" ]; then
    mkdir -p ./output/log
fi

if [ ! -d "./plugin_completion_log" ]; then
    mkdir -p ./plugin_completion_log
fi


# nohup bash scripts/rl/rl_multi.sh > rl_3072_fmt_orm_cst05_gen8_penalty_11_lr1e_6_bsz128_wokl.log 2>&1 &

export TIME_STAMP=${timestamp}
export NCCL_SOCKET_NTHREADS=2
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_BLOCKING_WAIT=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_TC=106
export NCCL_IB_SL=0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO

NCCL_DEBUG=INFO \
REWARD_API_ADDRESS=10.82.122.76 \
QWEN_API_PORT=8000 \
VQA_WEIGHT=1 \
FMT_WEIGHT=0.5 \
CODE_WEIGHT=0.1 \
MAX_PIXELS=3211264 \
deepspeed --hostfile=/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/hostfile_4013_2 \
    scripts/rl/rlhf_ds.py \
    --rlhf_type grpo \
    --model /mmu_mllm_hdd_2/yifanzhang/models/tool_final/qwen_tool_all_data_180k_alldata_wogemini_retool2k_mmeu10k_filter_tuihuo_lr1e_6/v0-20250618-155155/checkpoint-192 \
    --external_plugins /mmu_mllm_hdd_2/yifanzhang/agent_latest_code/examples/train/grpo/plugin/agent_rm.py \
    --reward_funcs fmt_orm vqa_orm cst_orm \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_max_model_len 10240 \
    --vllm_tensor_parallel_size 1 \
    --vllm_limit_mm_per_prompt '{"image": 10, "video": 0}' \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.8 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mllm_hdd/yfzhang/Agent-R1/construct_data/deepeyes/RL.jsonl \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --max_pixels 3211264 \
    --max_length 10240 \
    --max_completion_length 3072 \
    --stop_words "'</code>' '</answer>' '<|im_end|>'" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --padding_side left \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1, "num_cycles": 0.5}' \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --split_dataset_ratio 0 \
    --eval_steps 20000 \
    --save_steps 20 \
    --save_total_limit 100000 \
    --logging_steps 1 \
    --output_dir /mllm_hdd/yfzhang/Agent-R1/agent_latest_code/models/rl_3072_fmt_orm_cst05_gen8_penalty_11_lr1e_6_bsz128_wokl \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.1 \
    --deepspeed zero3 \
    --O3 true \
    --vllm_enforce_eager true \
    --log_completions true \
    --report_to tensorboard \
    --async_generate false \
    --num_iterations 1 \
    --overlong_filter true \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true \
    --attn_impl flash_attn \

    # --beta 0.001 \

# cd eval
# bash infer.sh

    # --dynamic_sample true \
    # --overlong_filter true \
    # --max_resample_times 3 \
# --move_model_batches 6 \
    # --move_model_batches 6 \
    # --sleep_level 1 \
# --sleep_level 1 \
# --max_grad_norm 0.5 \
# --epsilon_high 0.28 \
    # --lora_rank 128 \
    # --lora_alpha 256 \
    # --system 'prompt.txt' \

# --resume_from_checkpoint /mmu_mllm_hdd_2/yifanzhang/models/tool_final/RL_ckpt_orm_fmt_new/v5-20250627-140910/checkpoint-500 \

# NCCL_P2P_DISABLE=0 \
# NCCL_IB_DISABLE=0 \
# NCCL_IB_GID_INDEX=3 \
# NCCL_IB_HCA=mlx5 \
# NCCL_IB_QPS_PER_CONNECTION=4 \
# NCCL_IB_TIMEOUT=19 \
# NCCL_NET_OVERHEAD=1000 \
# NCCL_SOCKET_IFNAME=eth \
# NCCL_DEBUG=INFO \
# MODEL_API_ADDRESS="10.82.121.34,10.82.122.98,10.82.120.218" \
# no_proxy="10.48.52.87,10.82.122.104,10.82.121.28,11.40.77.1,11.40.76.193,11.40.75.102,11.40.75.131,11.40.78.114,11.43.22.118,11.40.74.136" \
# MODEL_API_PORT="8000" \
# MASTER_PORT=29600 \
# NPROC_PER_NODE=8 \
# MIN_PIXELS=4096 \
# MAX_PIXELS=3010560 \
# MAX_NUM=4 \
# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
# python main.py rlhf \
# --model /mmu_mllm_hdd_2/wenbin/SFT/Keye-8B/AutoThink/20250612.auto_think_total_tianke_v6.v1_add_visual_agent/output/v0-20250612-111534/checkpoint-6000 \
# --hostfile=/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/hostfile_2057 \
# MAX_PIXELS=3211264 \
# MASTER_PORT=38325 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# NPROC_PER_NODE=8 \
# REWARD_API_ADDRESS=10.82.121.22 \
# VQA_WEIGHT=1 \