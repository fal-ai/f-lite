from pathlib import Path
from typing import Optional

import torch
from jsonargparse import auto_cli
from rich.console import Console

from .pipeline import FLitePipeline

# Initialize rich console
console = Console()

def generate_images(
    prompt: str,
    output_file: str,
    model: str = "Freepik/F-Lite",
    negative_prompt: Optional[str] = None,
    seed: int = 0,
    guidance_scale: float = 3.5,
    steps: int = 30,
    width: int = 1344,
    height: int = 896,
    cpu_offload: bool = True,
    device: Optional[str] = None,
    num_images: int = 1,
):
    """
    Generate images using the F-Lite pipeline.
    
    Args:
        prompt: Text prompt for image generation
        output_file: Output filename for the generated image(s). If num_images > 1, the index will be appended (e.g., image.png -> image-1.png, image-2.png).
        model: HuggingFace model ID. By default, it uses the Freepik/F-Lite-test model.
        negative_prompt: Negative text prompt
        seed: Random seed for generation
        guidance_scale: Guidance scale for the diffusion process
        steps: Number of inference steps
        width: Width of the generated image
        height: Height of the generated image
        cpu_offload: Whether to offload models to CPU when not in use
        device: Device to use for inference (cuda or cpu)
        num_images: Number of images to generate
    """
    # Handle device selection
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            console.print("[yellow]Warning:[/yellow] CUDA device not found. Using CPU instead.")
    
    console.print(f"Using device: [bold blue]{device}[/bold blue]")
    
    # Set torch device
    torch_device = torch.device(device)
    
    # Load model
    console.print(f"Loading model: [bold green]{model}[/bold green]")

    # Trick required because it is not a native diffusers model
    from diffusers.pipelines.pipeline_loading_utils import (
        ALL_IMPORTABLE_CLASSES,
        LOADABLE_CLASSES,
    )
    LOADABLE_CLASSES["f_lite"] = LOADABLE_CLASSES["f_lite.model"] = {"DiT": ["save_pretrained", "from_pretrained"]}
    ALL_IMPORTABLE_CLASSES["DiT"] = ["save_pretrained", "from_pretrained"]

    pipe = FLitePipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    
    if cpu_offload:
        # Memory usage reduction
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(torch_device)
    
    # Enable optimizations
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # Generate image
    console.print(f"Generating {num_images} image(s) with prompt: [italic]{prompt}[/italic]")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=torch.Generator(device=torch_device).manual_seed(seed),
        num_images_per_prompt=num_images,
    )
    
    # Save image(s)
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, image in enumerate(output.images):
        if num_images == 1:
            current_output_path = output_path
        else:
            # Append index for multiple images, starting from 1 for the second image
            index_suffix = f"-{i}" if i > 0 else ""
            current_output_path = output_dir / f"{output_stem}{index_suffix}{output_suffix}"
        
        console.print(f"Saving image to: [bold green]{current_output_path}[/bold green]")
        image.save(current_output_path)
    
    console.print(f"[bold green]{len(output.images)} image(s) generated successfully![/bold green]")

if __name__ == "__main__":
    auto_cli(generate_images, as_positional=False) 