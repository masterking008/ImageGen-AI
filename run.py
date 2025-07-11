#!/usr/bin/env python3
"""
Simple run script for ImgGen AI
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.app import ImgGenApp

def main():
    print("🚀 Starting ImgGen AI...")
    
    try:
        # Initialize the application
        app = ImgGenApp()
        
        # Generate a test image
        print("📝 Generating test image...")
        result = app.generate_image(
            prompt="A beautiful sunset over mountains, digital art",
            width=512,
            height=512,
            steps=10  # Reduced for faster testing
        )
        
        if result["success"]:
            print(f"✅ Image generated: {result['image_path']}")
        else:
            print(f"❌ Generation failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()