 #!/bin/bash

# 设置环境变量
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用的GPU ID，根据实际情况修改

# 设置训练参数
NUM_GPUS=4  # 使用的GPU数量，根据实际情况修改
BATCH_SIZE=32  # 每个GPU的批次大小
EPOCHS=100
LOG_DIR="./logs"
CHECKPOINT_DIR="./checkpoints"
DATA_DIR="./data"  # 数据目录，根据实际情况修改

# 创建必要的目录
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# 创建一个唯一的会话名称
SESSION_NAME="mamba_train_$(date +%Y%m%d_%H%M%S)"

# 在screen中启动训练
screen -dmS $SESSION_NAME bash -c "
    echo '启动分布式训练...'
    torchrun --nproc_per_node=$NUM_GPUS train_distributed.py \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --data-dir $DATA_DIR \
        --log-dir $LOG_DIR \
        --checkpoint-dir $CHECKPOINT_DIR \
        --num-workers 8 \
        2>&1 | tee $LOG_DIR/training_output.log
    echo '训练完成'
"

echo "训练已在screen会话\"$SESSION_NAME\"中启动"
echo "使用以下命令查看训练状态:"
echo "  screen -r $SESSION_NAME"
echo "使用Ctrl+A, D分离screen会话"