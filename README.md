# ImgGen AI

A high-fidelity image generation and editing system designed to replicate and extend the capabilities of tools like DALL·E 3 and ChatGPT's image model — completely offline.

## 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Text-to-Image Generation** | Generate high-resolution, detailed images from simple natural language prompts |
| **Image-to-Image Transformation** | Modify or stylize existing images with new prompts or reference styles |
| **Inpainting / Editing** | Remove and regenerate parts of an image using a masked region and a new prompt |
| **Pose / Structure Control** | Use stick figures, edge maps, or depth maps to guide image structure using ControlNet |
| **Identity-Preserving Stylization** | Stylize faces or portraits while retaining identity using InstantID |
| **Style Transfer from Reference** | Apply the color and visual style of one image to another using IP-Adapter or LoRA modules |
| **No-Code Visual Workflow** | Build generation pipelines visually using ComfyUI's drag-and-drop interface |

## 🏗️ Architecture

The system combines open-source models:
- **Stable Diffusion XL** - Core image generation
- **ControlNet** - Structure and pose control
- **InstantID** - Identity preservation
- **LoRA** - Style adaptation
- **ComfyUI** - Visual workflow interface

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Launch ComfyUI interface
python src/main.py
```

## 📁 Project Structure

```
ImgGen-AI/
├── src/                    # Core application code
├── comfyui/               # ComfyUI integration
├── config/                # Configuration files
├── data/                  # Models and data storage
├── examples/              # Example workflows
└── docs/                  # Documentation
```

## 🔧 Installation

See [Installation Guide](docs/installation.md) for detailed setup instructions.

## 📖 Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Workflow Examples](docs/workflows.md)

## 🤝 Contributing

See [Contributing Guidelines](docs/contributing.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.# ImageGen-AI
