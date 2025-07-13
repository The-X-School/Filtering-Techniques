#!/bin/bash
# A100 Optimization Setup Script
# Run this before training for maximum performance

echo "Setting up A100 optimizations..."

# Set CUDA environment variables for A100
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Enable optimized CUDA kernels
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Optimize for A100 tensor cores
export NVIDIA_TF32_OVERRIDE=1

# Set optimal thread counts
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Install optimized packages if not already installed
echo "Installing/updating packages for A100 optimization..."
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade transformers>=4.36.0
pip install --upgrade peft>=0.7.0
pip install --upgrade accelerate>=0.24.0
pip install flash-attn --no-build-isolation

# Verify GPU and optimization setup
echo "Verifying A100 setup..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
print(f'PyTorch version: {torch.__version__}')
print(f'Flash Attention available: {hasattr(torch.nn.functional, \"scaled_dot_product_attention\")}')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}')
"

echo "A100 optimization setup complete!"
echo "Now run your training script with: python Train_model.py"
