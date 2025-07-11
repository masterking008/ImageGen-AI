"""
Image processing utilities
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Union, Tuple, Optional

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load image from file path"""
    try:
        image = Image.open(image_path)
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")

def save_image(image: Image.Image, output_path: Union[str, Path], quality: int = 95):
    """Save PIL Image to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        image.save(output_path, 'JPEG', quality=quality)
    else:
        image.save(output_path, 'PNG')

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """Resize image to target size"""
    if maintain_aspect:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)

def create_canny_edge(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Create Canny edge map from image"""
    # Convert PIL to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to PIL
    return Image.fromarray(edges).convert("RGB")

def create_depth_map(image: Image.Image) -> Image.Image:
    """Create depth map from image (placeholder - would use actual depth estimation model)"""
    # This is a placeholder - in practice, you'd use a depth estimation model
    # like MiDaS or DPT
    gray = image.convert("L")
    return gray.convert("RGB")

def extract_pose_keypoints(image: Image.Image) -> Image.Image:
    """Extract pose keypoints from image (placeholder)"""
    # This is a placeholder - in practice, you'd use OpenPose or similar
    # to extract human pose keypoints
    return image

def create_mask_from_bbox(image_size: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Create mask from bounding box coordinates"""
    mask = Image.new("L", image_size, 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask

def blend_images(base_image: Image.Image, overlay_image: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Blend two images with alpha transparency"""
    return Image.blend(base_image, overlay_image, alpha)

def preprocess_controlnet_image(image: Image.Image, control_type: str) -> Image.Image:
    """Preprocess image for ControlNet based on control type"""
    if control_type == "canny":
        return create_canny_edge(image)
    elif control_type == "depth":
        return create_depth_map(image)
    elif control_type == "pose":
        return extract_pose_keypoints(image)
    else:
        return image