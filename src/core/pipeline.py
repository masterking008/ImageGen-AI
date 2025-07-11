"""
Image Generation Pipeline
Handles different generation workflows: text-to-image, image-to-image, inpainting
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
from PIL import Image

from .model_manager import ModelManager
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.image_utils import load_image, save_image

class ImageGenerationPipeline:
    """Main pipeline for image generation tasks"""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load base models
        self.sdxl_pipeline = None
        self.controlnet_pipeline = None
        self.instantid_pipeline = None
        
    def _ensure_models_loaded(self, model_type: str):
        """Ensure required models are loaded"""
        if model_type == "sdxl" and self.sdxl_pipeline is None:
            self.sdxl_pipeline = self.model_manager.load_sdxl()
        elif model_type == "controlnet" and self.controlnet_pipeline is None:
            self.controlnet_pipeline = self.model_manager.load_controlnet()
        elif model_type == "instantid" and self.instantid_pipeline is None:
            self.instantid_pipeline = self.model_manager.load_instantid()
    
    def text_to_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text prompt"""
        self._ensure_models_loaded("sdxl")
        
        try:
            # Extract parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            steps = params.get("num_inference_steps", 20)
            guidance_scale = params.get("guidance_scale", 7.5)
            seed = params.get("seed", None)
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate image
            result = self.sdxl_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                return_dict=True
            )
            
            # Save image
            image = result.images[0]
            output_path = self._save_generated_image(image, "txt2img")
            
            return {
                "success": True,
                "image_path": str(output_path),
                "parameters": params
            }
            
        except Exception as e:
            self.logger.error(f"Text-to-image generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def image_to_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform existing image with new prompt"""
        self._ensure_models_loaded("sdxl")
        
        try:
            # Load input image
            image_path = params.get("image_path")
            input_image = load_image(image_path)
            
            # Extract parameters
            prompt = params.get("prompt", "")
            strength = params.get("strength", 0.8)
            guidance_scale = params.get("guidance_scale", 7.5)
            
            # Generate transformed image
            result = self.sdxl_pipeline(
                prompt=prompt,
                image=input_image,
                strength=strength,
                guidance_scale=guidance_scale,
                return_dict=True
            )
            
            # Save image
            image = result.images[0]
            output_path = self._save_generated_image(image, "img2img")
            
            return {
                "success": True,
                "image_path": str(output_path),
                "parameters": params
            }
            
        except Exception as e:
            self.logger.error(f"Image-to-image generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def inpaint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inpaint masked regions of an image"""
        self._ensure_models_loaded("sdxl")
        
        try:
            # Load input image and mask
            image_path = params.get("image_path")
            mask_path = params.get("mask_path")
            
            input_image = load_image(image_path)
            mask_image = load_image(mask_path).convert("L")
            
            # Extract parameters
            prompt = params.get("prompt", "")
            strength = params.get("strength", 1.0)
            
            # Generate inpainted image
            result = self.sdxl_pipeline(
                prompt=prompt,
                image=input_image,
                mask_image=mask_image,
                strength=strength,
                return_dict=True
            )
            
            # Save image
            image = result.images[0]
            output_path = self._save_generated_image(image, "inpaint")
            
            return {
                "success": True,
                "image_path": str(output_path),
                "parameters": params
            }
            
        except Exception as e:
            self.logger.error(f"Inpainting failed: {e}")
            return {"success": False, "error": str(e)}
    
    def controlnet_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with ControlNet guidance"""
        self._ensure_models_loaded("controlnet")
        
        # Implementation for ControlNet generation
        # This would handle pose, depth, canny edge control
        pass
    
    def instantid_stylize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stylize image while preserving identity"""
        self._ensure_models_loaded("instantid")
        
        # Implementation for InstantID stylization
        pass
    
    def _save_generated_image(self, image: Image.Image, prefix: str) -> Path:
        """Save generated image with timestamp"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        output_path = Path(self.config.get("output_dir", "data/outputs")) / filename
        
        save_image(image, output_path)
        return output_path