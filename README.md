<div align="center">

# F Lite

[**Simo Ryu**](https://x.com/cloneofsimo)&nbsp;&nbsp;&nbsp;&nbsp;
[**Lu Pengqi**](https://x.com/kuer5ord)&nbsp;&nbsp;&nbsp;&nbsp;
[**Javier Martín Juan**](https://x.com/info_libertas)&nbsp;&nbsp;&nbsp;&nbsp;
[**Iván de Prado Alonso**](https://x.com/ivanprado)&nbsp;&nbsp;&nbsp;&nbsp;
<br />

<a href="https://huggingface.co/spaces/Freepik/F-Lite"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Demo&color=orange"></a> &ensp;
<a href="https://huggingface.co/spaces/Freepik/F-Lite-Texture"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Texture%20Demo&color=orange"></a> &ensp;
<a href="#comfyui-nodes"><img src="https://img.shields.io/static/v1?label=%E2%9A%99%EF%B8%8F%20ComfyUI&message=Node&color=purple"></a> &ensp;
<a href="https://huggingface.co/Freepik/F-Lite"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Released&message=Model&color=green"></a> &ensp;
<a href="https://huggingface.co/Freepik/F-Lite-Texture"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Released&message=Texture%20Model&color=green"></a> &ensp;
<a href="assets/F Lite Technical Report.pdf"><img src="https://img.shields.io/static/v1?label=%F0%9F%93%84%20Technical&message=Report&color=darkred&logo=pdf"></a> &ensp;

</div>

<p align="center">
  <img src="assets/output_tight_mosaic.jpeg" alt="F Lite generated images mosaic" width="100%">
</p>

F Lite is a 10B parameter diffusion model created by [Freepik](https://www.freepik.com) and [Fal](https://fal.ai), trained exclusively on copyright-safe and SFW content. The model was trained on Freepik's internal dataset comprising approximately 80 million copyright-safe images, making it the first publicly available model of this scale trained exclusively on legally compliant and SFW content.

Read our [technical report](assets/F%20Lite%20Technical%20Report.pdf) for more details about the architecture and training process.

## Project updates

* 🎉 **April 29, 2025**: F Lite is released!

## Weights


| Model Version | Link | Description | Notes | HF Demo | Fal.ai Demo |
|---------------|------|-------------|-------|---------|------------|
| **Standard Model** | [Freepik/F-Lite](https://huggingface.co/Freepik/F-Lite) | Base model suitable for general-purpose image generation | - | [Demo](https://huggingface.co/spaces/Freepik/F-Lite) | [Fal.ai Demo](https://fal.ai/models/fal-ai/f-lite/standard) |
| **Texture Model** | [Freepik/F-Lite-Texture](https://huggingface.co/Freepik/F-Lite-Texture) | Specialized version with richer textures and enhanced details | • Requires more detailed prompts<br>• May be more prone to malformations<br>• Less effective for vector-style imagery | [Demo](https://huggingface.co/spaces/Freepik/F-Lite-Texture) | [Fal.ai Demo](https://fal.ai/models/fal-ai/f-lite/texture) |

## ComfyUI Nodes

F Lite can be used within ComfyUI for a more visual workflow experience. Two example workflows are provided:

### Simple Workflow (F-lite-simple.json)

This workflow doesn't require any additional extensions and provides basic F Lite image generation functionality.

- Load the workflow by importing `F-lite-simple.json` in ComfyUI
- Perfect for quick testing without additional setup

### Advanced Workflow with SuperPrompt (F-lite-superprompt.json)

This workflow is **recommended** as F Lite works significantly better with detailed, longer prompts. It uses the SuperPrompt feature to expand simple prompts into more detailed descriptions.

#### Requirements

This workflow requires the following ComfyUI extensions:
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) - Provides the `Superprompt` node for prompt expansion
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) - Provides the `ShowText` node for text visualization

To install these extensions:

```bash
cd [your ComfyUI folder]/custom_nodes
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts
git clone https://github.com/kijai/ComfyUI-KJNodes
```

After installing the extensions, restart ComfyUI and load the `F-lite-superprompt.json` workflow.

## Command Line and `diffusers`

### Installation

```bash
pip install -r requirements.txt
```

### Command line generation

You can generate images using the provided `generate.py` script:

```bash
python -m f_lite.generate \
  --prompt "A photorealistic landscape of a mountain lake at sunset with reflections in the water" \
  --output_file "generated_image.png" \
  --model "Freepik/F-Lite" \
  --width 1344 \
  --height 896 \
  --steps 30 \
  --guidance_scale 6 \
  --seed 42
```

### Diffusers

Here's a basic example of how to use the F Lite pipeline:

```python
import torch
from f_lite import FLitePipeline

# Trick required because it is not a native diffusers model
from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES, ALL_IMPORTABLE_CLASSES
LOADABLE_CLASSES["f_lite"] = LOADABLE_CLASSES["f_lite.model"] = {"DiT": ["save_pretrained", "from_pretrained"]}
ALL_IMPORTABLE_CLASSES["DiT"] = ["save_pretrained", "from_pretrained"]

pipeline = FLitePipeline.from_pretrained("Freepik/F-Lite", torch_dtype=torch.bfloat16)
pipeline.enable_model_cpu_offload() # For less memory consumption. Alternatively, pipeline.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate an image
output = pipeline(
    prompt="A photorealistic 3D render of a charming, mischievous young boy, approximately eight years old, possessing the endearingly unusual features of long, floppy donkey ears that droop playfully over his shoulders and a surprisingly small, pink pig nose that twitches slightly.  His eyes, a sparkling, intelligent hazel, are wide with a hint of playful mischief, framed by slightly unruly, sandy-brown hair that falls in tousled waves across his forehead.",
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=3.0,
    negative_prompt=None,
)

# Save the generated image
output.images[0].save("image.png")
```

The model requires a card with at least 24GB of VRAM to operate, although using quantization has not been explored and it is expected to reduce the memory footprint even more.

You can enable Adaptive Projected Guidance by setting the parameter `apg_config` to `APGConfig(enabled=True)`.

Using [negative prompts](NEGATIVE_PROMPT.md) can help improve the quality of the generated images.

## Fine-tuning and LoRAs

It is possible to fine-tune F Lite using your own data. Read the [Fine-tuning](FINE-TUNING.md) documentation for more information.

## Graphical User Interface

f-lite includes a Gradio-based GUI that provides an intuitive interface for image generation. 

![image](https://github.com/user-attachments/assets/3df36bdc-b2ea-4c10-a7a9-88eccd7548d5)


## Launching Gradio GUI

```
python f-lite-gradio-gui.py
```

### Gradio GUI Features
- Basic parameter configuration
- Augmented Parallel Guidance for more consistent  results
- Preset resolutions for common aspect ratios (square, portrait, landscape)
- Continuous generation mode with random seed iteration
- Dynamic prompt enhancement with wildcard support (`{option1|option2|option3}` syntax, as well as `__filename__` for .txt-file wildcards)
- Enhanced prompts using [SuperPrompt](https://huggingface.co/roborovski/superprompt-v1)
- Controls for randomization with SuperPrompt and wildcards while using fixed seeds
- Prompt prefix / suffix

## Acknowledgements

This model uses [T5 XXL](https://huggingface.co/google/t5-v1_1-xxl)and [Flux Schnell VAE](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## License

The F Lite weights are licensed under the permissive [CreativeML Open RAIL-M license](LICENSE). The T5 XXL and Flux Schnell VAE are licensed under Apache 2.0.

## Citation

If you find our work helpful, please cite it!

```
@article{ryu2025flite,
  title={F Lite Technical Report},
  author={Ryu, Simo and Pengqi, Lu and Mart\'in Juan, Javier and de Prado Alonso, Iv\'an},
  year={2025}
}
```
