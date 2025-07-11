"""
Core Application Class for ImgGen AI
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .pipeline import ImageGenerationPipeline
from .model_manager import ModelManager
from ..utils.config import Config
from ..utils.logger import get_logger

class ImgGenApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.logger = get_logger(__name__)
        self.config = Config(config_path)
        
        # Initialize core components
        self.model_manager = ModelManager(self.config)
        self.pipeline = ImageGenerationPipeline(self.model_manager, self.config)
        
        self.logger.info("ImgGen AI initialized successfully")
    
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: str = "",
                      width: int = 1024,
                      height: int = 1024,
                      steps: int = 20,
                      guidance_scale: float = 7.5,
                      **kwargs) -> Dict[str, Any]:
        """Generate image from text prompt"""
        
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            **kwargs
        }
        
        return self.pipeline.text_to_image(params)
    
    def transform_image(self,
                       image_path: str,
                       prompt: str,
                       strength: float = 0.8,
                       **kwargs) -> Dict[str, Any]:
        """Transform existing image with new prompt"""
        
        params = {
            "image_path": image_path,
            "prompt": prompt,
            "strength": strength,
            **kwargs
        }
        
        return self.pipeline.image_to_image(params)
    
    def inpaint_image(self,
                     image_path: str,
                     mask_path: str,
                     prompt: str,
                     **kwargs) -> Dict[str, Any]:
        """Inpaint masked regions of an image"""
        
        params = {
            "image_path": image_path,
            "mask_path": mask_path,
            "prompt": prompt,
            **kwargs
        }
        
        return self.pipeline.inpaint(params)
    
    def start_api_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Start REST API server"""
        from ..ui.api import create_api_app
        import uvicorn
        
        app = create_api_app(self)
        uvicorn.run(app, host=host, port=port)
    
    def start_cli(self):
        """Start command line interface"""
        from ..ui.cli import CLIInterface
        
        cli = CLIInterface(self)
        cli.run()