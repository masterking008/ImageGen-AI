# Installation Guide

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or Apple Silicon Mac
- 8GB+ RAM (16GB+ recommended)
- 20GB+ free disk space for models

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ImgGen-AI.git
cd ImgGen-AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Models

```bash
python scripts/download_models.py
```

### 5. Install ComfyUI (Optional)

For the visual workflow interface:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git comfyui
cd comfyui
pip install -r requirements.txt
cd ..
```

## Platform-Specific Setup

### NVIDIA GPU (CUDA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install xformers for memory efficiency
pip install xformers
```

### Apple Silicon (MPS)

```bash
# PyTorch with MPS support is included in requirements.txt
# No additional setup needed
```

### CPU Only

```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Configuration

1. Copy the default configuration:
```bash
cp config/default.yaml config/local.yaml
```

2. Edit `config/local.yaml` to customize settings:
   - Model paths
   - Output directories
   - Performance settings
   - UI preferences

## Verification

Test the installation:

```bash
python examples/basic_generation.py
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or enable CPU offload in config
2. **Model download fails**: Check internet connection and disk space
3. **Import errors**: Ensure virtual environment is activated

### Getting Help

- Check the [FAQ](faq.md)
- Open an issue on GitHub
- Join our Discord community