"""
Configuration Management
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for ImgGen AI"""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model_dir": "data/models",
            "output_dir": "data/outputs",
            "cache_dir": "data/cache",
            "temp_dir": "data/temp",
            
            "generation": {
                "default_width": 1024,
                "default_height": 1024,
                "default_steps": 20,
                "default_guidance_scale": 7.5,
                "max_width": 2048,
                "max_height": 2048,
                "max_steps": 100
            },
            
            "models": {
                "sdxl_model": "stabilityai/stable-diffusion-xl-base-1.0",
                "controlnet_models": {
                    "canny": "diffusers/controlnet-canny-sdxl-1.0",
                    "depth": "diffusers/controlnet-depth-sdxl-1.0",
                    "pose": "thibaud/controlnet-openpose-sdxl-1.0"
                }
            },
            
            "ui": {
                "host": "127.0.0.1",
                "port": 8188,
                "auto_launch": True
            },
            
            "logging": {
                "level": "INFO",
                "file": "logs/imggen.log"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, indent=2)