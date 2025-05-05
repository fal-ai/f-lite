import gradio as gr
import random
import re
from pathlib import Path
from PIL import Image
import torch
from f_lite.pipeline import FLitePipeline, APGConfig
from f_lite.generate import generate_images
import os
import threading
import time
import shutil
import subprocess

# --- Wildcard logic ---
WILDCARD_DIR = Path(__file__).parent / "wildcards"

# Check and create wildcards directory if it doesn't exist
if not WILDCARD_DIR.exists():
    WILDCARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created wildcards directory at: {WILDCARD_DIR}")

# Cache for wildcard files
_wildcard_cache = {}

# --- Dropdown resolutions ---
PRESET_RESOLUTIONS = [
    {"name": "[Square] 768×768 (1:1)",    "width": 768,  "height": 768},
    {"name": "[Square] 1024×1024 (1:1)",    "width": 1024,  "height": 1024},
    {"name": "[Square] 1536×1536 (1:1)",    "width": 1536,  "height": 1536},
    {"name": "[Portrait] 544×960 (9:16)",   "width": 544,  "height": 960},
    {"name": "[Portrait] 832×1248 (2:3)",  "width": 832,  "height": 1248},
    {"name": "[Portrait] 864×1536 (9:16)",   "width": 864, "height": 1536},
    {"name": "[Landscape] 960×544 (16:9)",   "width": 960,  "height": 544},
    {"name": "[Landscape] 1248×832 (3:2)",  "width": 1248, "height": 832},
    {"name": "[Landscape] 1536×864 (16:9)",   "width": 1536, "height": 864},
]

# For tracking the last generated image path
LAST_GENERATED_IMAGE_PATH = None

# Cancellation controller
class CancellationManager:
    def __init__(self):
        self.cancelled = False
        self._interrupt_event = threading.Event()
    
    def cancel(self):
        """Signal cancellation to all processes"""
        self.cancelled = True
        self._interrupt_event.set()
        print("Cancellation requested!")
    
    def reset(self):
        """Reset cancellation state"""
        self.cancelled = False
        self._interrupt_event.clear()
    
    def is_cancelled(self):
        """Check if cancellation was requested"""
        return self.cancelled
    
    def callback(self, step, timestep, latents):
        """Callback for diffusion pipeline to check for cancellation during steps"""
        if self.cancelled:
            # Raise a KeyboardInterrupt to stop the pipeline immediately
            raise KeyboardInterrupt("Generation cancelled by user")
        return {"latents": latents}

# Create global cancellation manager
cancel_manager = CancellationManager()

def get_random_line_from_file(file_path):
    if file_path not in _wildcard_cache:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            _wildcard_cache[file_path] = lines
        except Exception:
            return ""
    lines = _wildcard_cache.get(file_path, [])
    if not lines:
        return ""
    return random.choice(lines)

def find_wildcard_file(name):
    """Find a .txt file by name (case-insensitive) in WILDCARD_DIR or its subfolders."""
    for root, _, files in os.walk(WILDCARD_DIR):
        for file in files:
            if file.lower() == f"{name.lower()}.txt":
                return os.path.join(root, file)
    return None

def process_wildcards(text):
    # {a|b|c} wildcard
    def curly_replacer(match):
        options = match.group(1).split("|")
        return random.choice(options)
    text = re.sub(r"\{([^{}]+)\}", curly_replacer, text)
    # __myfile__ wildcard
    def file_replacer(match):
        filename = match.group(1)
        file_path = find_wildcard_file(filename)
        if file_path:
            return get_random_line_from_file(file_path)
        return match.group(0)
    text = re.sub(r"__([a-zA-Z0-9_\-\/]+)__", file_replacer, text)
    return text

# --- Image action functions ---
def open_output_folder():
    """Open the output folder in file explorer"""
    OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
    try:
        if os.path.exists(OUTPUT_ROOT):
            # Handle different platforms
            if os.name == 'nt':  # Windows
                os.startfile(OUTPUT_ROOT)
            elif os.name == 'posix':  # macOS/Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.call(['open', OUTPUT_ROOT])
                else:  # Linux
                    subprocess.call(['xdg-open', OUTPUT_ROOT])
            return f"Opening folder: {OUTPUT_ROOT}"
        else:
            return f"Output folder not found: {OUTPUT_ROOT}"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def save_image_to_downloads():
    """Provide a copy of the last generated image for download"""
    global LAST_GENERATED_IMAGE_PATH
    if LAST_GENERATED_IMAGE_PATH and os.path.exists(LAST_GENERATED_IMAGE_PATH):
        # Return the image path for Gradio to handle the download
        return LAST_GENERATED_IMAGE_PATH
    return None

# --- Pipeline singleton ---
PIPELINE = None
def get_pipeline(model="Freepik/F-Lite-Texture"):
    global PIPELINE
    if PIPELINE is None:
        if not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA (GPU) is not available! The app will not work efficiently.")
        # --- Required hack for F-Lite diffusers compatibility ---
        from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES, ALL_IMPORTABLE_CLASSES
        LOADABLE_CLASSES["f_lite"] = LOADABLE_CLASSES["f_lite.model"] = {"DiT": ["save_pretrained", "from_pretrained"]}
        ALL_IMPORTABLE_CLASSES["DiT"] = ["save_pretrained", "from_pretrained"]
        # --------------------------------------------------------
        PIPELINE = FLitePipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
        try:
            PIPELINE.vae.enable_slicing()
            PIPELINE.vae.enable_tiling()
            # Always use GPU if available
            if torch.cuda.is_available():
                # Enable CPU offload to reduce VRAM usage
                PIPELINE.enable_model_cpu_offload()
            else:
                PIPELINE.to("cpu")
        except Exception:
            pass
    return PIPELINE

# --- Gradio interface ---
def validate_dimensions(width, height):
    """Validate that dimensions are compatible with the model's requirements"""
    if width % 8 != 0 or height % 8 != 0:
        return False, f"Both width ({width}) and height ({height}) must be divisible by 8 for the model to work properly."
    return True, None

def round_dimension(val):
    divisor = 8
    try:
        val = int(val)
        rounded = int(round(val / divisor) * divisor)
        return rounded
    except Exception:
        return val

def set_cancel_flag():
    """Request cancellation of the current generation"""
    cancel_manager.cancel()
    # Return an immediate visual update to the UI to show we're cancelling
    return gr.update(value="Cancelling...", variant="secondary")

def generate(
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    width,
    height,
    seed,
    model,
    generate_mode,
    status_area,
    current_image
):
    global LAST_GENERATED_IMAGE_PATH
    
    # If we already have an image, don't clear it until new one is ready
    starting_image = None if current_image is None else current_image
    
    # Reset cancellation state and update UI
    cancel_manager.reset()
    
    # Validate dimensions
    is_valid, error_msg = validate_dimensions(width, height)
    if not is_valid:
        return (
            starting_image,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            f"Error: {error_msg}\n\nPlease adjust the dimensions and try again."
        )
    
    # Format the status message with all generation parameters
    status_msg = f"Model: {model}\n"
    status_msg += f"Resolution: {width}×{height}\n"
    status_msg += f"Steps: {steps}, CFG: {guidance_scale}\n"
    if seed == -1:
        seed_value = random.randint(0, 2**32 - 1)
        status_msg += f"Seed: {seed_value} (random)\n"
    else:
        seed_value = seed
        status_msg += f"Seed: {seed_value}\n"
    
    # Truncate prompts if too long for display
    prompt_display = prompt if len(prompt) < 100 else prompt[:97] + "..."
    neg_prompt_display = negative_prompt if len(negative_prompt) < 100 else negative_prompt[:97] + "..."
    
    status_msg += f"Prompt: {prompt_display}\n"
    if negative_prompt:
        status_msg += f"Negative prompt: {neg_prompt_display}\n"
    
    # Python command format with proper quote escaping
    escaped_prompt = prompt.replace('"', '\\"')
    escaped_negative = negative_prompt.replace('"', '\\"') if negative_prompt else None
    
    status_msg += f"\npython generate.py --model {model} --width {width} --height {height} --steps {steps} --cfg {guidance_scale} --seed {seed_value}"
    if escaped_negative:
        status_msg += f' --negative "{escaped_negative}"'
    status_msg += f' "{escaped_prompt}"'
    
    # Update UI to show generation status and hide/show necessary buttons
    yield starting_image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), status_msg
    
    # Round dimensions to nearest multiple of 8
    width = round_dimension(width)
    height = round_dimension(height)
    from datetime import datetime
    import os

    OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")

    def save_image(image):
        global LAST_GENERATED_IMAGE_PATH
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%Y-%m-%d - %H-%M-%S")
        batch_index = "001"
        output_dir = os.path.join(OUTPUT_ROOT, date_str)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{time_str}-{batch_index}.png")
        image.save(out_path)
        LAST_GENERATED_IMAGE_PATH = out_path
        return out_path

    # This function is now a generator for Gradio streaming
    def make_divisible(value, divisor=8):
        return int(round(value / divisor) * divisor)

    def single_generation(seed_value, width, height):
        # Check if cancelled before starting
        if cancel_manager.is_cancelled():
            print("Generation cancelled before starting")
            return None
            
        # Wildcard processing
        processed_prompt = process_wildcards(prompt)
        processed_negative_prompt = process_wildcards(negative_prompt) if negative_prompt else None
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_device == "cpu":
            print("Warning: CUDA is not available. Using CPU instead.")
        
        # Enforce width and height to be divisible by 8
        adj_width = make_divisible(width, 8)
        adj_height = make_divisible(height, 8)
        if (adj_width != width or adj_height != height):
            print(f"Adjusted resolution to {adj_width}x{adj_height} to match model requirements (divisible by 8).")
        width = adj_width
        height = adj_height
        
        try:
            pipe = get_pipeline(model)
            
            # Clear CUDA cache before generation to help avoid fragmentation/OOM
            if torch_device == "cuda":
                torch.cuda.empty_cache()
                
            generator = torch.Generator(device=torch_device).manual_seed(seed_value)
            
            # Check cancellation again before starting generation
            if cancel_manager.is_cancelled():
                print("Generation cancelled before pipeline call")
                return None
                
            # Register our callback to the pipeline for step-by-step cancellation checks
            output = pipe(
                prompt=processed_prompt,
                negative_prompt=processed_negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=1,
                callback=cancel_manager.callback,  # Add callback for cancellation
                callback_steps=1  # Check every step
            )
            
            # Check if cancelled during generation
            if cancel_manager.is_cancelled():
                print("Generation was cancelled during process")
                return None
                
            image = output.images[0] if hasattr(output, "images") else output[0]
            save_image(image)
            return image
            
        except KeyboardInterrupt:
            print("Generation interrupted by cancellation")
            return None
        except Exception as e:
            print(f"Error during generation: {e}")
            return None
        finally:
            # OOM fix: delete pipeline and clear CUDA cache after each generation
            if 'pipe' in locals():
                del pipe
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        if generate_mode == "forever" and seed == -1:
            # Generate forever until cancelled
            iteration = 0
            while not cancel_manager.is_cancelled():
                iteration += 1
                print(f"Starting generation #{iteration} (forever mode)")
                current_seed = random.randint(0, 2**32 - 1)
                
                # Update the status with the new seed for this iteration
                current_status = status_msg.replace(f"Seed: {seed_value} (random)", f"Seed: {current_seed} (random, iteration {iteration})")
                
                try:
                    image = single_generation(current_seed, width, height)
                    
                    # Stop if cancelled during generation
                    if cancel_manager.is_cancelled() or image is None:
                        print("Generation loop cancelled")
                        break
                    
                    # Update status to show completion
                    completion_status = current_status.replace("Starting generation:", f"Image generated (#{iteration}):")
                    
                    yield image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), completion_status
                    
                    # Small delay to allow UI updates and cancellation
                    for _ in range(5):  # Check cancel flag multiple times during delay
                        if cancel_manager.is_cancelled():
                            print("Cancelled during wait period")
                            break
                        time.sleep(0.1)  # Split delay into smaller chunks
                        
                except Exception as e:
                    print(f"Error in generation loop: {e}")
                    if not cancel_manager.is_cancelled():
                        cancel_manager.cancel()  # Force cancellation on error
                    break
        else:
            # Single generation mode
            if seed == -1:
                current_seed = random.randint(0, 2**32 - 1)
                # Update status with actual seed value
                status_msg = status_msg.replace(f"Seed: {seed_value} (random)", f"Seed: {current_seed} (random)")
            else:
                current_seed = seed
            
            print(f"Starting single generation with seed {current_seed}")
            image = single_generation(current_seed, width, height)
            
            if not cancel_manager.is_cancelled() and image is not None:
                # Update status to show completion
                completion_status = status_msg.replace("Starting generation:", "Image generated:")
                yield image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), completion_status
            else:
                completion_status = status_msg.replace("Starting generation:", "Generation cancelled:")
                yield starting_image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), completion_status
                print("Single generation was cancelled or failed")
    except Exception as e:
        print(f"Error during generation process: {e}")
    finally:
        # Always reset UI state when finished or cancelled
        print("Generation finished or cancelled, resetting UI")
        yield gr.update(), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update()

def set_resolution(res):
    return gr.update(value=res[0]), gr.update(value=res[1])

def build_interface():
    with gr.Blocks(title="f-lite Gradio GUI") as demo:
        # Main 2-column layout
        with gr.Row():
            # LEFT COLUMN: all controls/settings (50% width)
            with gr.Column(scale=1, min_width=400):
                # Top row: Two separate buttons and a cancel button
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate-btn",
                        scale=1,
                        visible=True
                    )
                    generate_forever_btn = gr.Button(
                        "Generate Forever",
                        elem_id="generate-forever-btn",
                        scale=1,
                        visible=True
                    )
                    cancel_btn = gr.Button(
                        "Cancel After Current Generation",
                        elem_id="cancel-btn",
                        scale=1,
                        visible=False,
                        variant="stop"
                    )
                # Prompt fields (full width)
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    elem_id="prompt-textbox"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    value="",
                    elem_id="negative-prompt-textbox"
                )
                # Resolution row
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=[preset["name"] for preset in PRESET_RESOLUTIONS],
                        label="Resolution",
                        value=PRESET_RESOLUTIONS[0]["name"] if PRESET_RESOLUTIONS else None,
                        elem_id="resolution-dropdown"
                    )
                    width = gr.Number(
                        label="Width",
                        value=PRESET_RESOLUTIONS[0]["width"] if PRESET_RESOLUTIONS else 1344,
                        precision=0,
                        elem_id="width-field"
                    )
                    height = gr.Number(
                        label="Height",
                        value=PRESET_RESOLUTIONS[0]["height"] if PRESET_RESOLUTIONS else 896,
                        precision=0,
                        elem_id="height-field"
                    )
                # Steps, CFG, Seed row
                with gr.Row():
                    steps = gr.Slider(
                        5, 100, value=30, step=1, label="Steps", scale=1,
                        elem_id="steps-slider"
                    )
                    guidance_scale = gr.Slider(
                        1, 20, value=6, step=0.1, label="CFG", scale=1,
                        elem_id="cfg-slider"
                    )
                    seed = gr.Number(
                        label="Seed (-1=random)",
                        value=-1,
                        precision=0,
                        scale=1,
                        elem_id="seed-field"
                    )
                # Model picker
                model = gr.Dropdown(
                    choices=["Freepik/F-Lite", "Freepik/F-Lite-Texture"],
                    label="Model",
                    value="Freepik/F-Lite-Texture",
                    elem_id="model-dropdown"
                )
            # RIGHT COLUMN: output image, status area, and action buttons (50% width)
            with gr.Column(scale=1):
                # Image output area
                output = gr.Image(
                    label="Generated Image",
                    elem_id="output-image",
                    interactive=False,
                    show_download_button=False
                )
                
                # Status area
                status_area = gr.Textbox(
                    label="Status",
                    elem_id="status-area",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
                
                # Image action buttons (centered)
                with gr.Row(elem_id="image-actions"):
                    # Open output folder button
                    open_folder_btn = gr.Button(
                        "📁 Open Output Folder",
                        elem_id="open-folder-btn",
                        scale=1
                    )
        
        # Hidden state for generate mode
        generate_mode = gr.State("single")
        
        # Button logic - Image actions
        open_folder_btn.click(
            fn=open_output_folder,
            inputs=[],
            outputs=[]
        )
        
        # Button logic - Cancel Generation
        cancel_btn.click(
            fn=set_cancel_flag,
            inputs=[],
            outputs=[cancel_btn],
        )
        
        # Button logic - Single Generation
        generate_btn.click(
            lambda: "single",
            outputs=[generate_mode]
        ).then(
            generate,
            inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed, model, generate_mode, status_area, output],
            outputs=[output, generate_btn, generate_forever_btn, cancel_btn, status_area],
        )
        
        # Button logic - Generate Forever
        generate_forever_btn.click(
            lambda: "forever",
            outputs=[generate_mode]
        ).then(
            generate,
            inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed, model, generate_mode, status_area, output],
            outputs=[output, generate_btn, generate_forever_btn, cancel_btn, status_area],
        )
        
        # Resolution dropdown change handler
        preset_dropdown.change(
            fn=lambda x: next(((p["width"], p["height"]) for p in PRESET_RESOLUTIONS if p["name"] == x), (None, None)),
            inputs=[preset_dropdown],
            outputs=[width, height]
        )
        
        # --- Hotkey and style ---
        gr.HTML("""
<style>
#generate-btn button, #generate-btn {
    background-color: #d35400 !important;
    color: white !important;
    border: none !important;
    font-size: 1.2em !important;
    font-weight: bold !important;
    width: 100% !important;
    min-width: 120px;
}
#generate-forever-btn button, #generate-forever-btn {
    background-color: #27ae60 !important;
    color: white !important;
    border: none !important;
    font-size: 1.2em !important;
    font-weight: bold !important;
    width: 100% !important;
    min-width: 120px;
}
#cancel-btn button, #cancel-btn {
    background-color: #e74c3c !important;
    color: white !important;
    border: none !important;
    font-size: 1.2em !important;
    font-weight: bold !important;
    width: 100% !important;
    min-width: 120px;
}
#image-actions {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 20px;
}
#open-folder-btn, #save-image-btn, #delete-image-btn {
    flex: 0 0 auto;
    min-width: 160px !important;
    max-width: 200px !important;
}
#status-area textarea {
    font-family: monospace !important;
    white-space: pre-wrap !important;
}
#resolution-dropdown {
    flex: 1 1 0% !important;
    min-width: 120px !important;
}
#width-field, #height-field {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    display: flex !important;
    align-items: center !important;
}
#width-field input, #height-field input {
    width: 16ch !important;
    min-width: 16ch !important;
    max-width: 20ch !important;
    text-align: center !important;
}

</style>
<script>
    // Add mouseover (title) tooltips to controls and their children
    window.addEventListener('DOMContentLoaded', function() {
        const tooltips = [
            ["generate-btn", "Click to generate a single image with the current settings."],
            ["generate-forever-btn", "Click to continuously generate new images with random seeds until stopped."],
            ["cancel-btn", "Click to cancel the current generation process."],
            ["prompt-textbox", "Describe what you want to see in the image. Use detailed, descriptive language for best results."],
            ["negative-prompt-textbox", "Describe things you do NOT want in the image. This helps the model avoid unwanted features."],
            ["resolution-dropdown", "Choose a preset resolution for the generated image."],
            ["width-field", "Width of the generated image in pixels. Typical values: 1024–1536."],
            ["height-field", "Height of the generated image in pixels. Typical values: 896–1536."],
            ["steps-slider", "How many denoising steps to use. Higher values = more detail, but slower. Default: 30."],
            ["cfg-slider", "Classifier-Free Guidance scale. Higher = more faithful to prompt, but may reduce creativity."],
            ["seed-field", "Random seed for reproducibility. Use -1 to pick a new random seed each time."],
            ["model-dropdown", "Choose which F-Lite model to use. 'Texture' gives more detailed results."],
            ["open-folder-btn", "Open the output folder where generated images are saved."],
            ["save-image-btn", "Download the currently displayed image to your computer."],
            ["delete-image-btn", "Delete the current image from the output folder (requires confirmation)."],
            ["status-area", "Displays generation parameters and status information."]
        ];
        for (const [id, tip] of tooltips) {
            const container = document.getElementById(id);
            if (container) {
                // Try to set title on all relevant children (input, select, textarea, label)
                const tagNames = ["input", "select", "textarea", "label", "button", "div"];
                let found = false;
                for (const tag of tagNames) {
                    const els = container.getElementsByTagName(tag);
                    for (const el of els) {
                        el.title = tip;
                        found = true;
                    }
                }
                // If nothing found, set on container
                if (!found) container.title = tip;
            }
        }
    });
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            let btn = document.querySelector('#generate-btn button') || document.querySelector('#generate-btn');
            if (btn && window.getComputedStyle(btn).display !== 'none') btn.click();
        }
        // Add Escape key to cancel generation
        if (e.key === 'Escape') {
            let btn = document.querySelector('#cancel-btn button') || document.querySelector('#cancel-btn');
            if (btn && window.getComputedStyle(btn).display !== 'none') btn.click();
        }
    });
</script>
""")
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
