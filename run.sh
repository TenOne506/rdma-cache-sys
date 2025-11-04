#!/bin/bash

# 运行 Web 界面（使用 Conda 环境）

echo "Starting RDMA Cache Simulation Web Interface..."

cd web || { echo "Failed to enter 'web' directory"; exit 1; }

# 初始化 Conda（使 conda 命令可用）
eval "$(conda shell.bash hook)"

# 设置 Conda 环境名称
ENV_NAME="rdma-cache-env"

# 检查 Conda 环境是否存在
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating Conda environment: $ENV_NAME..."
    # 从 requirements.txt 创建环境（假设是 pip 格式）
    # 如果你有 environment.yml，建议改用它
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# 激活 Conda 环境
conda activate "$ENV_NAME"

# 安装依赖（使用 pip 安装 requirements.txt 中的包）
pip install -r ../requirements.txt

# 设置 Flask 环境变量
export FLASK_APP=app.py
export FLASK_ENV=development  # 注意：Flask 2.3+ 已弃用 FLASK_ENV，建议用 FLASK_DEBUG

# 运行 Flask 应用
python app.py