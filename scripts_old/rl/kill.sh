#!/bin/bash

# 获取所有vllm进程的PID
pids=$(pgrep infer_model.py)

# 检查是否有vllm进程在运行
if [ -z "$pids" ]; then
  echo "没有找到vllm进程"
else
  # 杀死所有vllm进程
  echo "杀死以下vllm进程: $pids"
  kill -9 $pids
  echo "所有vllm进程已被杀死"
fi