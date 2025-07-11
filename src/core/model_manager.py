"""
Model Manager - Handles loading and management of AI models
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from diffusers import StableDiffusionXLPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline

from ..utils.config import Config
from ..utils.logger import get_logger

class ModelManager:
    """Manages loading and caching of AI models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.device = self._get_device()
        self.models = {}
        
        # Model paths
        self.model_dir = Path(config.get("model_dir", "data/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_sdxl(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0") -> StableDiffusionXLPipeline:
        """Load Stable Diffusion XL pipeline"""
        if "sdxl" in self.models:
            return self.models["sdxl"]
        
        self.logger.info(f"Loading SDXL model: {model_id}")
        
        try:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device != "cpu" else None
            )
            
            pipeline = pipeline.to(self.device)
            
            # Enable memory efficient attention
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                pipeline.enable_xformers_memory_efficient_attention()
            
            # Enable CPU offload for memory efficiency
            if self.device == "cuda":
                pipeline.enable_model_cpu_offload()
            
            self.models["sdxl"] = pipeline
            self.logger.info("SDXL model loaded successfully")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to load SDXL model: {e}")
            raise
    
    def load_controlnet(self, controlnet_type: str = "canny") -> StableDiffusionXLControlNetPipeline:
        """Load ControlNet pipeline"""
        cache_key = f"controlnet_{controlnet_type}"
        
        if cache_key in self.models:
            return self.models[cache_key]
        
        self.logger.info(f"Loading ControlNet model: {controlnet_type}")
        
        try:
            # Load ControlNet model
            controlnet_models = {
                "canny": "diffusers/controlnet-canny-sdxl-1.0",
                "depth": "diffusers/controlnet-depth-sdxl-1.0",
                "pose": "thibaud/controlnet-openpose-sdxl-1.0"
            }
            
            controlnet_id = controlnet_models.get(controlnet_type, controlnet_models["canny"])
            
            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            
            # Load SDXL pipeline with ControlNet
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device != "cpu" else None
            )
            
            pipeline = pipeline.to(self.device)
            
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                pipeline.enable_xformers_memory_efficient_attention()
            
            self.models[cache_key] = pipeline
            self.logger.info(f"ControlNet {controlnet_type} loaded successfully")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to load ControlNet {controlnet_type}: {e}")
            raise
    
    def load_instantid(self) -> Any:
        """Load InstantID pipeline for identity preservation"""
        if "instantid" in self.models:
            return self.models["instantid"]
        
        self.logger.info("Loading InstantID model")
        
        try:
            # InstantID implementation would go here
            # This is a placeholder for the actual InstantID integration
            self.logger.warning("InstantID not yet implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load InstantID: {e}")
            raise
    
    def load_lora(self, lora_path: str, adapter_name: str = "default") -> bool:
        """Load LoRA adapter"""
        try:
            if "sdxl" not in self.models:
                self.load_sdxl()
            
            pipeline = self.models["sdxl"]
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            
            self.logger.info(f"LoRA adapter '{adapter_name}' loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load LoRA adapter: {e}")
            return False
    
    def unload_model(self, model_key: str):
        """Unload a specific model to free memory"""
        if model_key in self.models:
            del self.models[model_key]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.logger.info(f"Model '{model_key}' unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "device": self.device,
            "loaded_models": list(self.models.keys()),
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            }
        return {"cpu_only": True}