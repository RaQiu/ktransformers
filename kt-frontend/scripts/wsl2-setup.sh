#!/bin/bash
# =============================================================================
# ktransformers WSL2 一键安装脚本
# 用法: bash wsl2-setup.sh
# =============================================================================
set -e

echo "=========================================="
echo " ktransformers WSL2 环境安装"
echo "=========================================="

# ---- 1. 系统依赖 ----
echo ""
echo "[1/6] 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake ninja-build \
    libnuma-dev libhwloc-dev pkg-config curl wget

# ---- 2. CUDA Toolkit (WSL2 不需要装驱动，只装 toolkit) ----
echo ""
echo "[2/6] 检查 CUDA..."
if command -v nvcc &> /dev/null; then
    echo "CUDA 已安装: $(nvcc --version | grep release)"
else
    echo "安装 CUDA Toolkit 12.9..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-9
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    echo "CUDA 安装完成: $(nvcc --version | grep release)"
fi

# 验证 GPU 可见
echo ""
echo "验证 GPU..."
nvidia-smi || { echo "错误: nvidia-smi 无法运行，请确认 Windows NVIDIA 驱动版本 >= 535"; exit 1; }

# ---- 3. Miniconda ----
echo ""
echo "[3/6] 安装 Miniconda..."
if command -v conda &> /dev/null; then
    echo "Conda 已安装"
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "Conda 安装完成"
fi

eval "$(conda shell.bash hook 2>/dev/null || $HOME/miniconda3/bin/conda shell.bash hook)"

# 创建环境
echo ""
echo "创建 Python 3.11 环境..."
conda create -n kt python=3.11 -y 2>/dev/null || true
conda activate kt

# ---- 4. PyTorch ----
echo ""
echo "[4/6] 安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# 验证
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ---- 5. ktransformers (kt-kernel) ----
echo ""
echo "[5/6] 安装 ktransformers..."
KT_DIR="$HOME/ktransformers"

if [ -d "$KT_DIR" ]; then
    echo "ktransformers 目录已存在，更新..."
    cd "$KT_DIR" && git pull
else
    git clone https://github.com/kvcache-ai/ktransformers.git "$KT_DIR"
    cd "$KT_DIR"
fi

# 安装 kt-kernel
cd "$KT_DIR/kt-kernel"
pip install -e .

# 验证
kt version

# ---- 6. SGLang (kvcache-ai fork) ----
echo ""
echo "[6/6] 安装 SGLang..."
SGLANG_DIR="$HOME/sglang"

if [ -d "$SGLANG_DIR" ]; then
    echo "sglang 目录已存在，更新..."
    cd "$SGLANG_DIR" && git pull
else
    git clone https://github.com/kvcache-ai/sglang.git "$SGLANG_DIR"
fi

cd "$SGLANG_DIR"
pip install -e "python"

echo ""
echo "=========================================="
echo " 安装完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate kt"
echo "  kt run --model-path <模型路径>"
echo ""
echo "模型存放建议:"
echo "  WSL2 可以访问 Windows 磁盘: /mnt/d/models"
echo "  例如: kt run --model-path /mnt/d/models/DeepSeek-V3"
echo ""
echo "提示: Windows 上的 D:\\models 在 WSL2 里是 /mnt/d/models"
echo ""
