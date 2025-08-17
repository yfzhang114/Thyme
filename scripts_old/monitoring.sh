#!/bin/bash
# nohup bash monitoring.sh > monitoring.out 2>&1 &

# 检查进程是否在运行的函数
# 检查任意一个进程是否在运行的函数
check_process_running() {
    pgrep -f train_dpo.py > /dev/null || pgrep -f train_dpo_mixup.py > /dev/null
    return $?
}


# 主循环
while true; do
    # 检查 train_dpo.py 是否在运行
    if check_process_running; then
        echo "train_dpo.py is running. Checking again in 5 minutes..."
        sleep 180  # 等待 5 分钟
    else
        nohup bash scripts/alignment/alignment_llavaov05b_multinode_2.sh >> llava_ov_05b_mmrlhf_v2_bsz196_ls0_5e_7_margin1.log 2>&1 &
        break  # 运行完脚本后退出循环
    fi
done

