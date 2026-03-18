#!/bin/bash

# ==========================================
# 1. 彻底解决 502 代理劫持问题
# ==========================================
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export WANDB_MODE=offline

# ==========================================
# 2. 环境变量与路径配置
# ==========================================
ROOT_DIR="/data/vepfs/users/intern/ruijie.sang/Code/WMNavigation/"
CONDA_PATH="/data/vepfs/users/intern/ruijie.sang/miniconda3/etc/profile.d/conda.sh"
NUM_GPU=1
INSTANCES=50
NUM_EPISODES_PER_INSTANCE=20
MAX_STEPS_PER_EPISODE=40
TASK="ObjectNav"
DATASET="hm3d_v0.2"
CFG="WMNav"
NAME="wmnav-Qwen2_5-VL-7B-Instruct-hm3dv2"
PROJECT_NAME="WMNav"
VENV_NAME="wmnav"
GPU_LIST=(0)
SLEEP_INTERVAL=200
LOG_FREQ=1
PORT=2000
CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --dataset ${DATASET}"

# 创建日志文件夹保存输出
LOG_DIR="/data/nas/users/ruijie.sang/logs_train"
mkdir -p "${LOG_DIR}"
echo "All outputs will be saved to ${LOG_DIR}"

# ==========================================
# 3. 启动 Aggregator
# ==========================================
echo "Starting Aggregator Session..."
(
    unset VIRTUAL_ENV
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v openpi | grep -v '.venv' | tr '\n' ':' | sed 's/:$//')

    source "${CONDA_PATH}"
    conda deactivate 2>/dev/null
    conda activate "${VENV_NAME}"
    cd "${ROOT_DIR}"

    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
    unset __EGL_VENDOR_LIBRARY_FILENAMES
    unset EGL_VISIBLE_DEVICES

    python scripts/aggregator.py --name "${TASK}_${NAME}" --project "${PROJECT_NAME}" --sleep "${SLEEP_INTERVAL}" --config "${CFG}" --port "${PORT}"
) > "${LOG_DIR}/aggregator_${NAME}.log" 2>&1 &

AGGREGATOR_PID=$!
echo "Aggregator PID: ${AGGREGATOR_PID}"

# 给 Aggregator 几秒钟的启动时间
sleep 5

# ==========================================
# 4. 进程清理函数
# ==========================================
INSTANCE_PIDS=()

cleanup() {
    echo -e "\nCaught interrupt signal (Ctrl+C). Cleaning up background processes..."
    
    for pid in "${INSTANCE_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            echo "Killed instance PID: $pid"
        fi
    done
    
    if kill -0 $AGGREGATOR_PID 2>/dev/null; then
        kill -9 $AGGREGATOR_PID
        echo "Killed aggregator PID: $AGGREGATOR_PID"
    fi
    exit 1
}

trap cleanup SIGINT SIGTERM

# ==========================================
# 5. 启动各个实例 (EGL 在 conda activate 之后设置)
# ==========================================
for instance_id in $(seq 0 $((INSTANCES - 1))); do
    GPU_ID=${GPU_LIST[$((instance_id % ${#GPU_LIST[@]}))]}
    echo "Starting Instance ${instance_id} on GPU ${GPU_ID}..."
    
    (
        unset VIRTUAL_ENV
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v openpi | grep -v '.venv' | tr '\n' ':' | sed 's/:$//')

        source "${CONDA_PATH}"
        conda deactivate 2>/dev/null
        conda activate "${VENV_NAME}"
        cd "${ROOT_DIR}"

        export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
        unset __EGL_VENDOR_LIBRARY_FILENAMES
        export EGL_VISIBLE_DEVICES=${GPU_ID}
        CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD} --instance ${instance_id}
    ) > "${LOG_DIR}/instance_${instance_id}.log" 2>&1 &
    
    PID=$!
    INSTANCE_PIDS+=($PID)
    echo "Instance ${instance_id} PID: ${PID}"
done

# ==========================================
# 6. 监控任务进度
# ==========================================
echo "Running... Waiting for all instances to finish."

for pid in "${INSTANCE_PIDS[@]}"; do
    wait $pid
done

echo "DONE: All instances have finished."

# ==========================================
# 7. 任务结束，发送终止信号
# ==========================================
echo "$(date): Sending termination signal to aggregator."
curl --noproxy "*" -X POST http://localhost:${PORT}/terminate
if [ $? -eq 0 ]; then
    echo "$(date): Termination signal sent successfully."
else
    echo "$(date): Failed to send termination signal."
fi

sleep 10
if kill -0 $AGGREGATOR_PID 2>/dev/null; then
    kill -9 $AGGREGATOR_PID
    echo "Killed aggregator PID: $AGGREGATOR_PID"
fi