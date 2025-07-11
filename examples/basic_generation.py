#!/usr/bin/env python3
"""
Basic Image Generation Example
Demonstrates text-to-image generation with ImgGen AI
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.app import ImgGenApp

def main():
    # Initialize the application
    app = ImgGenApp()
    
    # Example 1: Basic text-to-image generation
    print("Generating image from text prompt...")
    result = app.generate_image(
        prompt="A majestic dragon flying over a medieval castle at sunset, highly detailed, fantasy art",
        negative_prompt="blurry, low quality, distorted",
        width=1024,
        height=1024,
        steps=25,
        guidance_scale=7.5
    )
    
    if result["success"]:
        print(f"✓ Image generated successfully: {result['image_path']}")
    else:
        print(f"✗ Generation failed: {result['error']}")
    
    # Example 2: Generate with specific seed for reproducibility
    print("\nGenerating reproducible image with seed...")
    result = app.generate_image(
        prompt="A serene Japanese garden with cherry blossoms, zen, peaceful",
        seed=42,
        steps=20
    )
    
    if result["success"]:
        print(f"✓ Reproducible image generated: {result['image_path']}")
    else:
        print(f"✗ Generation failed: {result['error']}")

if __name__ == "__main__":
    main()