#!/usr/bin/env python3
"""
Model Download Script
Downloads required models for ImgGen AI
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.logger import setup_logger

def download_model(repo_id: str, local_dir: Path, logger):
    """Download model from Hugging Face Hub"""
    try:
        logger.info(f"Downloading {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"✓ Downloaded {repo_id}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {repo_id}: {e}")
        return False

def main():
    logger = setup_logger("model_downloader")
    config = Config()
    
    model_dir = Path(config.get("model_dir", "data/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Models to download
    models = [
        {
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "local_path": model_dir / "sdxl-base"
        },
        {
            "repo_id": "stabilityai/stable-diffusion-xl-refiner-1.0", 
            "local_path": model_dir / "sdxl-refiner"
        },
        {
            "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
            "local_path": model_dir / "controlnet-canny"
        },
        {
            "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
            "local_path": model_dir / "controlnet-depth"
        },
        {
            "repo_id": "thibaud/controlnet-openpose-sdxl-1.0",
            "local_path": model_dir / "controlnet-pose"
        }
    ]
    
    logger.info("Starting model downloads...")
    
    success_count = 0
    for model in models:
        if download_model(model["repo_id"], model["local_path"], logger):
            success_count += 1
    
    logger.info(f"Download complete: {success_count}/{len(models)} models downloaded successfully")
    
    if success_count < len(models):
        logger.warning("Some models failed to download. Check your internet connection and try again.")
        sys.exit(1)
    else:
        logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    main()