"""
ComfyUI Integration and Interface Launch
"""

import subprocess
import sys
from pathlib import Path
from ..utils.logger import get_logger

def launch_interface(host: str = "127.0.0.1", port: int = 8188):
    """Launch ComfyUI interface"""
    logger = get_logger(__name__)
    
    try:
        # Check if ComfyUI is installed
        comfyui_path = Path("comfyui")
        
        if not comfyui_path.exists():
            logger.error("ComfyUI not found. Please install ComfyUI first.")
            logger.info("Run: git clone https://github.com/comfyanonymous/ComfyUI.git comfyui")
            return False
        
        # Launch ComfyUI
        cmd = [
            sys.executable,
            str(comfyui_path / "main.py"),
            "--listen", host,
            "--port", str(port)
        ]
        
        logger.info(f"Launching ComfyUI at http://{host}:{port}")
        subprocess.run(cmd, cwd=comfyui_path)
        
    except KeyboardInterrupt:
        logger.info("ComfyUI shutdown requested")
    except Exception as e:
        logger.error(f"Failed to launch ComfyUI: {e}")
        return False
    
    return True