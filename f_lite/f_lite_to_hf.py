import time
from contextlib import contextmanager
from pathlib import Path

import torch
from jsonargparse import auto_cli
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .pt import load_f_lite_pt

# Initialize rich console
console = Console()


@contextmanager
def track_progress(description, title=None):
    """
    Context manager to track progress of a long-running operation.

    Args:
        description: Description of the task being performed
        title: Optional title for the panel

    Yields:
        A progress object that can be used to update the task
    """
    if title:
        console.print(f"[bold]{title}[/bold]")

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}..."),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)
        yield progress, task
        progress.update(task, completed=True)

    elapsed_time = time.time() - start_time
    console.print(f"{description} took: [bold yellow]{elapsed_time:.2f}[/bold yellow] seconds")


def make_tensors_contiguous(pipe):
    """Make all model tensors contiguous before saving."""
    console.print(Panel("Making tensors contiguous before saving...", title="Preparing Model"))
    for model in [pipe.dit_model, pipe.vae, pipe.text_encoder]:
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()


def f_lite_to_hf(
    input_checkpoint: Path,
    output_dir: Path,
    generate_test_image: bool = True,
    test_prompt: str = "A photorealistic 3D render of a charming, mischievous young boy, approximately eight years old, possessing the endearingly unusual features of long, floppy donkey ears that droop playfully over his shoulders and a surprisingly small, pink pig nose that twitches slightly.",
    test_image_height: int = 1024,
    test_image_width: int = 1024,
    test_num_inference_steps: int = 30,
    test_guidance_scale: float = 3.0,
    test_seed: int = 43,
):
    """
    Convert F-Lite checkpoint from PT format to Hugging Face format.
    
    Args:
        input_checkpoint: Path to the input checkpoint in PT format
        output_dir: Path to save the converted model in Hugging Face format
        generate_test_image: Whether to generate a test image to verify the model works
        test_prompt: Test prompt for image generation
        test_image_height: Test image height
        test_image_width: Test image width
        test_num_inference_steps: Number of inference steps for test image generation
        test_guidance_scale: Guidance scale for test image generation
        test_seed: Random seed for test image generation
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(Panel(f"Using device: [bold blue]{device}[/bold blue]", title="Device Info"))

    # Load the model from PT format
    with track_progress("Loading model", f"Loading model from [bold green]{input_checkpoint}[/bold green]") as (
        progress,
        task,
    ):
        pipe = load_f_lite_pt(
            model_path=input_checkpoint,
            device=device,
            enable_vae_slicing=True
        )
        pipe.to(torch_dtype=torch.bfloat16)
        pipe.enable_vae_slicing()

    # Generate a test image if requested
    if generate_test_image:
        console.print(
            Panel(
                f"Generating test image with prompt: [italic]{test_prompt}[/italic]", title="Test Image Generation"
            )
        )
        output = pipe(
            prompt=test_prompt,
            negative_prompt=None,
            height=test_image_height,
            width=test_image_width,
            num_inference_steps=test_num_inference_steps,
            guidance_scale=test_guidance_scale,
            generator=torch.Generator(device=device).manual_seed(test_seed),
        )

        # Save the generated image
        output_path = output_dir / "test_image.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.images[0].save(output_path)
        console.print(f"Test image saved to: [bold green]{output_path}[/bold green]")

    # Make tensors contiguous before saving
    make_tensors_contiguous(pipe)

    # Save the model in regular precision
    with track_progress("Saving model", f"Saving model to [bold green]{output_dir}[/bold green] in bf16 format") as (
        progress,
        task,
    ):
        pipe.save_pretrained(output_dir)

    console.print(Panel("[bold green]Conversion completed successfully![/bold green]", title="Success"))


if __name__ == "__main__":
    auto_cli(f_lite_to_hf, as_positional=False) 