#!/usr/bin/env python3
"""
Simple ImgGen AI Demo
"""

import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path
import yaml

def load_config():
    """Load configuration"""
    config_path = Path("config/default.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "output_dir": "data/outputs",
        "generation": {
            "default_width": 1024,
            "default_height": 1024,
            "default_steps": 20
        }
    }

def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    print("🚀 ImgGen AI - Simple Demo")
    
    config = load_config()
    device = get_device()
    print(f"📱 Using device: {device}")
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("📥 Loading Stable Diffusion XL...")
        
        # Load pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device != "cpu" else None
        )
        
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers enabled")
            except:
                print("⚠️  XFormers not available")
        
        print("🎨 Generating image...")
        
        # Generate image
        prompt = "A majestic dragon flying over a medieval castle at sunset, highly detailed, fantasy art"
        
        image = pipe(
            prompt=prompt,
            width=512,  # Smaller for faster generation
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        # Save image
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"generated_{timestamp}.png"
        image.save(output_path)
        
        print(f"✅ Image saved: {output_path}")
        print(f"📝 Prompt: {prompt}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try: pip install xformers (optional for better performance)")

if __name__ == "__main__":
    main()