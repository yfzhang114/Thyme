exp_name=SFT-5-2
model_name=Qwen2.5-32B-Instruct
save_name=SFT-5-2
data_file1=sft_146_d_rl_2.jsonl
# data_file2=sft_v27_10k.jsonl

HARDWARE=gpu

MY_HOME=/mnt/bn/sbl-yg
# MY_HOME=/mnt/bn/sbl-hl

attn_impl=flash_attn
if [ $HARDWARE == "ascend" ]; then
    attn_impl=sdpa
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi


sudo chown -R tiger $MY_HOME
pip3 install $MY_HOME/downloads/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip3 install deepspeed==0.14.5
pip3 install -e .

dataset1_path=$(realpath $MY_HOME/data/$data_file1)
# dataset2_path=$(realpath $MY_HOME/data/$data_file2)
model_path=$(realpath $MY_HOME/models/${model_name})
save_path=$(realpath $MY_HOME/ckpt/${save_name})


WANDB_PROJECT=yf_swift_r1reward \
    MASTER_ADDR=${ARNOLD_WORKER_0_HOST} MASTER_PORT=${ARNOLD_WORKER_0_PORT} NODE_RANK=${ARNOLD_ID} \
    no_proxy="" NNODES=${ARNOLD_WORKER_NUM} NPROC_PER_NODE=${ARNOLD_WORKER_GPU} \
    swift sft \
    --model $model_path \
    --loss_scale last_round \
    --train_type full \
    --dataset $dataset1_path \
    --split_dataset_ratio 0 \
    --attn_impl $attn_impl \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --deepspeed zero3_offload \
    --gradient_accumulation_steps 2 \
    --save_strategy epoch \
    --save_steps 1 \
    --save_total_limit 100 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir $save_path \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --run_name ${exp_name} \
    --report_to wandb