import gradio as gr
import random
import re
from pathlib import Path
from PIL import Image
import torch
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not found. SuperPrompt functionality will be disabled.")
    print("Please install it: pip install transformers")

from f_lite.pipeline import FLitePipeline, APGConfig
from f_lite.generate import generate_images
import os
import threading
import time
import shutil
import subprocess
from datetime import datetime

# --- Wildcard logic ---
WILDCARD_DIR = Path(__file__).parent / "wildcards"

if not WILDCARD_DIR.exists():
    WILDCARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created wildcards directory at: {WILDCARD_DIR}")

_wildcard_cache = {}

# --- SuperPrompt globals ---
_superprompt_tokenizer = None
_superprompt_model = None
_superprompt_initialized_attempted = False

def get_superprompt_components():
    global _superprompt_tokenizer, _superprompt_model, _superprompt_initialized_attempted
    
    if not TRANSFORMERS_AVAILABLE:
        if not _superprompt_initialized_attempted:
            print("SuperPrompt cannot be loaded because 'transformers' library is missing.")
            _superprompt_initialized_attempted = True
        return None, None

    if _superprompt_tokenizer is not None and _superprompt_model is not None:
        return _superprompt_tokenizer, _superprompt_model

    if not _superprompt_initialized_attempted:
        _superprompt_initialized_attempted = True 
        try:
            print("Loading SuperPrompt model and tokenizer...")
            _superprompt_tokenizer = T5Tokenizer.from_pretrained("roborovski/superprompt-v1")
            _superprompt_model = T5ForConditionalGeneration.from_pretrained("roborovski/superprompt-v1", device_map="auto")
            if hasattr(_superprompt_model, 'config') and _superprompt_model.config.model_type == 't5':
                print("SuperPrompt model and tokenizer loaded successfully.")
            else:
                print("SuperPrompt model loaded but appears to be non-functional. Disabling SuperPrompt.")
                _superprompt_tokenizer = None 
                _superprompt_model = None
        except Exception as e:
            print(f"Error loading SuperPrompt model/tokenizer: {e}")
            print("SuperPrompt functionality will be disabled.")
            _superprompt_tokenizer = None 
            _superprompt_model = None
            
    return _superprompt_tokenizer, _superprompt_model

# --- Dropdown resolutions ---
PRESET_RESOLUTIONS = [
    {"name": "[Square] 1024×1024 (1:1)",    "width": 1024,  "height": 1024},
    {"name": "[Square] 1216×1216 (1:1)",    "width": 1216, "height": 1216},
    {"name": "[Square] 1536×1536 (1:1)",    "width": 1536,  "height": 1536},
    {"name": "[Portrait] 640×960 (2:3)",    "width": 640, "height": 960},
    {"name": "[Portrait] 832×1248 (2:3)",   "width": 832, "height": 1248},
    {"name": "[Portrait] 864×1536 (9:16)",  "width": 864, "height": 1536},
    {"name": "[Portrait] 896×1600 (14:25)", "width": 896, "height": 1600},
    {"name": "[Landscape] 960×640 (3:2)",   "width": 960,  "height": 640},
    {"name": "[Landscape] 1248×832 (3:2)",  "width": 1248, "height": 832},
    {"name": "[Landscape] 1536×864 (16:9)", "width": 1536, "height": 864},
    {"name": "[Landscape] 1600×896 (25:14)","width": 1600, "height": 896},
]

LAST_GENERATED_IMAGE_PATH = None

class CancellationManager:
    def __init__(self):
        self.cancelled = False
        self._interrupt_event = threading.Event()
    
    def cancel(self):
        self.cancelled = True
        self._interrupt_event.set()
        print("Cancellation requested!")
    
    def reset(self):
        self.cancelled = False
        self._interrupt_event.clear()
    
    def is_cancelled(self):
        return self.cancelled
    
    def callback(self, step, timestep, latents):
        if self.cancelled:
            raise KeyboardInterrupt("Generation cancelled by user")
        return {"latents": latents}

cancel_manager = CancellationManager()

def get_random_line_from_file(file_path, seed=None):
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
    if seed is not None:
        rng = random.Random(seed)
        return rng.choice(lines)
    return random.choice(lines)

def find_wildcard_file(name):
    for root, _, files in os.walk(WILDCARD_DIR):
        for file in files:
            if file.lower() == f"{name.lower()}.txt":
                return os.path.join(root, file)
    return None

def process_wildcards(text, seed=None):
    rng = random.Random(seed) if seed is not None else random
    def curly_replacer(match):
        options = match.group(1).split("|")
        return rng.choice(options)
    text = re.sub(r"\{([^{}]+)\}", curly_replacer, text)
    def file_replacer(match):
        filename = match.group(1)
        file_path = find_wildcard_file(filename)
        if file_path:
            return get_random_line_from_file(file_path, seed)
        return match.group(0)
    text = re.sub(r"__([a-zA-Z0-9_\-\/]+)__", file_replacer, text)
    return text

# --- Image action functions ---
def open_output_folder():
    OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
    try:
        if os.path.exists(OUTPUT_ROOT):
            if os.name == 'nt':
                os.startfile(OUTPUT_ROOT)
            elif os.name == 'posix':
                if os.uname().sysname == 'Darwin':
                    subprocess.call(['open', OUTPUT_ROOT])
                else:
                    subprocess.call(['xdg-open', OUTPUT_ROOT])
            return f"Opening folder: {OUTPUT_ROOT}"
        else:
            return f"Output folder not found: {OUTPUT_ROOT}"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def save_image_to_downloads():
    global LAST_GENERATED_IMAGE_PATH
    if LAST_GENERATED_IMAGE_PATH and os.path.exists(LAST_GENERATED_IMAGE_PATH):
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
        from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES, ALL_IMPORTABLE_CLASSES
        LOADABLE_CLASSES["f_lite"] = LOADABLE_CLASSES["f_lite.model"] = {"DiT": ["save_pretrained", "from_pretrained"]}
        ALL_IMPORTABLE_CLASSES["DiT"] = ["save_pretrained", "from_pretrained"]
        PIPELINE = FLitePipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
        try:
            PIPELINE.vae.enable_slicing()
            PIPELINE.vae.enable_tiling()
            if torch.cuda.is_available():
                PIPELINE.enable_model_cpu_offload()
            else:
                PIPELINE.to("cpu")
        except Exception:
            pass
    return PIPELINE

# --- Gradio interface ---
def validate_dimensions(width, height):
    if width % 8 != 0 or height % 8 != 0:
        return False, f"Both width ({width}) and height ({height}) must be divisible by 8."
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
    cancel_manager.cancel()
    return gr.update(value="Cancelling...", variant="secondary")

def generate(
    prompt,
    negative_prompt,
    enhance_prompt,
    force_shuffle_enhance_prompt,
    force_shuffle_wildcards,
    enhance_prompt_length,
    steps,
    guidance_scale,
    width,
    height,
    seed,
    model,
    generate_mode,
    status_area,
    current_image,
    apg_enabled,
    apg_orthogonal_threshold,
    prompt_prefix,
    prompt_suffix
):
    global LAST_GENERATED_IMAGE_PATH
    
    starting_image = None if current_image is None else current_image
    cancel_manager.reset()
    
    is_valid, error_msg = validate_dimensions(width, height)
    if not is_valid:
        return (
            starting_image,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            f"Error: {error_msg}\n\nPlease adjust the dimensions and try again."
        )
    
    # Initialize prompt for generation
    base_prompt = prompt
    actual_prompt_for_generation = prompt
    enhancement_info_for_status = ""
    wildcard_info_for_status = ""

    # Seed handling
    seed_for_display_in_command = seed if seed != -1 else random.randint(0, 2**32 - 1)
    current_run_seed_info = f"Seed: {seed_for_display_in_command}\n"

    # Process wildcards with potentially forced random seed
    wildcard_seed = random.randint(0, 2**32 - 1) if force_shuffle_wildcards else seed_for_display_in_command
    processed_prompt = process_wildcards(base_prompt, wildcard_seed)
    processed_negative_prompt = process_wildcards(negative_prompt, wildcard_seed) if negative_prompt else None
    actual_prompt_for_generation = processed_prompt
    wildcard_info_for_status = " (Wildcards processed)"

    # Apply SuperPrompt enhancement if enabled
    if enhance_prompt:
        sp_tokenizer, sp_model_instance = get_superprompt_components()
        if sp_tokenizer and sp_model_instance:
            print("Attempting to enhance prompt with SuperPrompt...")
            input_text_for_superprompt = f"Expand the following prompt to add more detail: {processed_prompt}"
            try:
                target_device_for_inputs = sp_model_instance.device
                input_ids = sp_tokenizer(input_text_for_superprompt, return_tensors="pt").input_ids.to(target_device_for_inputs)
                
                # Use random seed for enhancement if forced
                if force_shuffle_enhance_prompt:
                    enhance_seed = random.randint(0, 2**32 - 1)
                    outputs = sp_model_instance.generate(input_ids, max_new_tokens=enhance_prompt_length, do_sample=True, top_k=50)
                else:
                    outputs = sp_model_instance.generate(input_ids, max_new_tokens=enhance_prompt_length)
                
                enhanced_prompt_text = sp_tokenizer.decode(outputs[0], skip_special_tokens=True)

                if enhanced_prompt_text and enhanced_prompt_text.strip() and \
                   enhanced_prompt_text.strip().lower() not in ["<pad>", "</s>", "pad", "eos"]:
                    print(f"Original prompt (after wildcards): {processed_prompt}")
                    print(f"Enhanced prompt: {enhanced_prompt_text}")
                    actual_prompt_for_generation = enhanced_prompt_text
                    enhancement_info_for_status = " (Enhanced by SuperPrompt)"
                else:
                    print("SuperPrompt returned empty or placeholder text. Using wildcard-processed prompt.")
                    enhancement_info_for_status = ""
            except Exception as e:
                print(f"Error during SuperPrompt enhancement: {e}. Using wildcard-processed prompt.")
                enhancement_info_for_status = f" (SuperPrompt error: {str(e)[:50]}...)"
        else:
            if TRANSFORMERS_AVAILABLE:
                enhancement_info_for_status = ""
            else:
                enhancement_info_for_status = " (SuperPrompt unavailable - 'transformers' lib missing)"

    # Save the enhanced prompt before applying prefix/suffix
    enhanced_prompt = actual_prompt_for_generation
    
    # Apply prefix and suffix if provided
    has_prefix_suffix = False
    final_prompt_with_prefix_suffix = actual_prompt_for_generation
    
    if prompt_prefix and prompt_prefix.strip():
        final_prompt_with_prefix_suffix = f"{prompt_prefix.strip()} {final_prompt_with_prefix_suffix}"
        has_prefix_suffix = True
        
    if prompt_suffix and prompt_suffix.strip():
        final_prompt_with_prefix_suffix = f"{final_prompt_with_prefix_suffix} {prompt_suffix.strip()}"
        has_prefix_suffix = True
        
    # Use the final prompt (with prefix/suffix) for generation
    if has_prefix_suffix:
        actual_prompt_for_generation = final_prompt_with_prefix_suffix

    # Construct status message
    status_msg = f"Model: {model}\n"
    status_msg += f"Resolution: {width}×{height}\n"
    status_msg += f"Steps: {steps}, CFG: {guidance_scale}\n"
    status_msg += current_run_seed_info
    if apg_enabled:
        status_msg += f"APG: Enabled (Threshold: {apg_orthogonal_threshold})\n"
    else:
        status_msg += f"APG: Disabled\n"
    status_msg += f"Shuffle Wildcards: {force_shuffle_wildcards}\n"
    status_msg += f"Shuffle Enhance Prompt: {force_shuffle_enhance_prompt}\n\n"
    
    # Display prompt stages
    has_wildcards = "{" in prompt or "__" in prompt
    was_enhanced = (actual_prompt_for_generation != processed_prompt and 
                   actual_prompt_for_generation != prompt)
    
    # Always show raw prompt
    status_msg += f"Raw prompt:\n {prompt}\n\n"
    
    # Show processed prompt only if wildcards were actually processed
    if has_wildcards and processed_prompt != prompt:
        status_msg += f"Processed prompt:\n {processed_prompt}\n\n"
    
    # Show enhanced prompt only if enhancement succeeded
    if was_enhanced:
        status_msg += f"Enhanced prompt:\n {enhanced_prompt}\n"

    # Show prefix/suffix prompt if applied
    if has_prefix_suffix:
        status_msg += f"\nPrefix / Suffix Fixed Prompt:\n {final_prompt_with_prefix_suffix}\n"
    
    # Show negative prompt if present
    if negative_prompt:
        status_msg += f"\nNegative prompt: {negative_prompt}"
        if has_wildcards and processed_negative_prompt != negative_prompt:
            status_msg += f"\nProcessed negative prompt: {processed_negative_prompt}"
    
    # Construct command string
    escaped_final_prompt = actual_prompt_for_generation.replace('"', '\"')
    escaped_negative = processed_negative_prompt.replace('"', '\"') if processed_negative_prompt else None
    
    now_for_command = datetime.now()
    date_str_for_command = now_for_command.strftime("%Y-%m-%d")
    time_str_for_command = now_for_command.strftime("%Y-%m-%d - %H-%M-%S")
    
    py_command_parts = [
        'Running command: \npy -m f_lite.generate',
        f'--prompt "{escaped_final_prompt}"',
        f'--output_file "output/{date_str_for_command}/{time_str_for_command}-001.png"',
        f'--model {model}',
        f'--width {width}',
        f'--height {height}',
        f'--steps {steps}',
        f'--guidance_scale {guidance_scale}',
        f'--seed {seed_for_display_in_command}'
    ]
    if escaped_negative:
        py_command_parts.append(f'--negative_prompt "{escaped_negative}"')
    
    py_command_string = " ".join(py_command_parts)
    status_msg += f"\n{py_command_string}"
    
    yield starting_image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), status_msg
    
    width = round_dimension(width)
    height = round_dimension(height)

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

    def make_divisible(value, divisor=8):
        return int(round(value / divisor) * divisor)

    def single_generation(seed_to_use, width_for_gen, height_for_gen, current_apg_enabled_for_gen, current_apg_orthogonal_threshold_for_gen, current_prompt, current_negative_prompt):
        if cancel_manager.is_cancelled():
            print("Generation cancelled before starting")
            return None
            
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_device == "cpu":
            print("Warning: CUDA is not available. Using CPU instead.")
        
        adj_width = make_divisible(width_for_gen, 8)
        adj_height = make_divisible(height_for_gen, 8)
        if (adj_width != width_for_gen or adj_height != height_for_gen):
            print(f"Adjusted resolution to {adj_width}x{adj_height} to match model requirements.")
        width_for_gen = adj_width
        height_for_gen = adj_height
        
        try:
            pipe = get_pipeline(model)
            apg_config_obj = APGConfig(enabled=current_apg_enabled_for_gen, orthogonal_threshold=current_apg_orthogonal_threshold_for_gen)
            
            if torch_device == "cuda":
                torch.cuda.empty_cache()
                
            generator = torch.Generator(device=torch_device).manual_seed(seed_to_use)
            
            if cancel_manager.is_cancelled():
                print("Generation cancelled before pipeline call")
                return None
                
            output = pipe(
                prompt=current_prompt,
                negative_prompt=current_negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                width=width_for_gen,
                height=height_for_gen,
                generator=generator,
                num_images_per_prompt=1,
                apg_config=apg_config_obj,
                callback=cancel_manager.callback,
                callback_steps=1
            )
            
            image = output.images[0] if hasattr(output, "images") else output[0]
            save_image(image)
            
            if cancel_manager.is_cancelled():
                print("Generation was cancelled during process")
                return image
            
            return image
            
        except KeyboardInterrupt:
            print("Generation interrupted by cancellation")
            return None
        except Exception as e:
            print(f"Error during generation: {e}")
            return None
        finally:
            if 'pipe' in locals():
                del pipe
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        if generate_mode == "forever" and seed == -1:
            iteration = 0
            cached_prompt = actual_prompt_for_generation
            cached_negative_prompt = processed_negative_prompt
            while not cancel_manager.is_cancelled():
                iteration += 1
                print(f"Starting generation #{iteration} (forever mode)")
                current_seed = random.randint(0, 2**32 - 1)
                
                # Process wildcards for this iteration
                current_processed_prompt = process_wildcards(base_prompt, current_seed) if force_shuffle_wildcards else processed_prompt
                current_negative_prompt = process_wildcards(negative_prompt, current_seed) if negative_prompt and force_shuffle_wildcards else processed_negative_prompt
                wildcard_info_for_status = " (Wildcards processed, shuffled)" if force_shuffle_wildcards else " (Wildcards processed, fixed)"

                # Apply SuperPrompt enhancement for this iteration
                current_prompt = current_processed_prompt  # Default to processed prompt
                if enhance_prompt and force_shuffle_enhance_prompt:
                    sp_tokenizer, sp_model_instance = get_superprompt_components()
                    if sp_tokenizer and sp_model_instance:
                        print("Re-enhancing prompt with SuperPrompt for iteration...")
                        input_text_for_superprompt = f"Expand the following prompt to add more detail: {current_processed_prompt}"
                        try:
                            target_device_for_inputs = sp_model_instance.device
                            input_ids = sp_tokenizer(input_text_for_superprompt, return_tensors="pt").input_ids.to(target_device_for_inputs)
                            outputs = sp_model_instance.generate(input_ids, max_new_tokens=enhance_prompt_length, do_sample=True, top_k=50)
                            enhanced_prompt_text = sp_tokenizer.decode(outputs[0], skip_special_tokens=True)

                            if enhanced_prompt_text and enhanced_prompt_text.strip() and \
                               enhanced_prompt_text.strip().lower() not in ["<pad>", "</s>", "pad", "eos"]:
                                print(f"Iteration prompt (after wildcards): {current_prompt}")
                                print(f"Iteration enhanced prompt: {enhanced_prompt_text}")
                                current_prompt = enhanced_prompt_text
                                enhancement_info_for_status = " (Enhanced by SuperPrompt, shuffled)"
                            else:
                                print("SuperPrompt returned empty or placeholder text. Using iteration wildcard-processed prompt.")
                                enhancement_info_for_status = " (SuperPrompt enhancement ineffective, using wildcard-processed)"
                        except Exception as e:
                            print(f"Error during SuperPrompt enhancement: {e}. Using iteration wildcard-processed prompt.")
                            enhancement_info_for_status = f" (SuperPrompt error: {str(e)[:50]}..., using wildcard-processed)"
                    else:
                        enhancement_info_for_status = " (SuperPrompt not loaded or unavailable, using wildcard-processed)"
                else:
                    current_prompt = cached_prompt
                    enhancement_info_for_status = " (Enhanced by SuperPrompt, fixed)" if enhance_prompt else " (No SuperPrompt enhancement)"

                # Save the pre-prefix/suffix prompt
                enhanced_iteration_prompt = current_prompt
                
                # Apply prefix and suffix for this iteration
                iteration_final_prompt = current_prompt
                has_prefix_suffix_for_iteration = False

                if prompt_prefix and prompt_prefix.strip():
                    iteration_final_prompt = f"{prompt_prefix.strip()} {iteration_final_prompt}"
                    has_prefix_suffix_for_iteration = True
                    
                if prompt_suffix and prompt_suffix.strip():
                    iteration_final_prompt = f"{iteration_final_prompt} {prompt_suffix.strip()}"
                    has_prefix_suffix_for_iteration = True
                
                if has_prefix_suffix_for_iteration:
                    current_prompt = iteration_final_prompt
                
                # Update status message for this iteration
                status_lines = status_msg.split('\n')
                updated_status_lines = []
                seed_line_found = False
                for line in status_lines:
                    if line.startswith("Seed:"):
                        updated_status_lines.append(f"Seed: {current_seed}")
                        seed_line_found = True
                    elif line.startswith("Prompt:") or line.startswith("Original Prompt:"):
                        continue  # Skip old prompt lines
                    else:
                        updated_status_lines.append(line)
                if not seed_line_found:
                    updated_status_lines.append(f"Seed: {current_seed}")
                
                # Add new prompt info
                is_successfully_enhanced = (enhanced_iteration_prompt != current_processed_prompt and enhance_prompt)
                
                if is_successfully_enhanced:
                    # Only add this part if enhanced
                    truncated_enhanced_prompt = enhanced_iteration_prompt if len(enhanced_iteration_prompt) < 100 else enhanced_iteration_prompt[:97] + "..."
                    updated_status_lines.append(f"Enhanced prompt:\n {truncated_enhanced_prompt}")
                else:
                    # Just show the processed prompt with appropriate info
                    truncated_prompt = current_processed_prompt if len(current_processed_prompt) < 100 else current_processed_prompt[:97] + "..."
                    updated_status_lines.append(f"Prompt: {truncated_prompt}{wildcard_info_for_status}")
                
                if has_prefix_suffix_for_iteration:
                    truncated_iteration_final_prompt = iteration_final_prompt if len(iteration_final_prompt) < 100 else iteration_final_prompt[:97] + "..."
                    updated_status_lines.append(f"Prefix / Suffix Fixed Prompt:\n {truncated_iteration_final_prompt}")
                
                if negative_prompt:
                    neg_prompt_display_truncated = current_negative_prompt if len(current_negative_prompt) < 100 else current_negative_prompt[:97] + "..."
                    updated_status_lines.append(f"Negative prompt: {neg_prompt_display_truncated}\n")
                
                current_iteration_status_msg = '\n'.join(updated_status_lines)
                
                try:
                    image = single_generation(current_seed, width, height, apg_enabled, apg_orthogonal_threshold, current_prompt, current_negative_prompt)
                    
                    if cancel_manager.is_cancelled() or image is None:
                        print("Generation loop cancelled")
                        break
                    
                    completion_status = current_iteration_status_msg.replace("Starting generation:", f"Image generated (#{iteration}):")
                    
                    yield image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), completion_status
                    
                    for _ in range(5):
                        if cancel_manager.is_cancelled():
                            print("Cancelled during wait period")
                            break
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error in generation loop: {e}")
                    if not cancel_manager.is_cancelled():
                        cancel_manager.cancel()
                    break
        else:
            current_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
            print(f"Starting single generation with seed {current_seed}")
            
            status_lines = status_msg.split('\n')
            updated_status_lines = []
            seed_line_found = False
            for line in status_lines:
                if line.startswith(f"Seed: {seed_for_display_in_command}"):
                    updated_status_lines.append(f"Seed: {current_seed}")
                    seed_line_found = True
                else:
                    updated_status_lines.append(line)
            status_msg = '\n'.join(updated_status_lines)
            
            # Apply prefix and suffix for single generation (if not already applied in first part)
            if has_prefix_suffix:
                # Already applied above
                pass
                
            image = single_generation(current_seed, width, height, apg_enabled, apg_orthogonal_threshold, actual_prompt_for_generation, processed_negative_prompt)
            
            if not cancel_manager.is_cancelled() and image is not None:
                completion_status = status_msg.replace("Starting generation:", "Image generated:")
                yield image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), completion_status
            else:
                completion_status = status_msg.replace("Starting generation:", "Generation cancelled:")
                yield starting_image, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="Cancel After Current Generation"), completion_status
                print("Single generation was cancelled or failed")
    except Exception as e:
        print(f"Error during generation process: {e}")
    finally:
        print("Generation finished\n")
        yield (starting_image if image is None else image), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update()

def set_resolution(res):
    return gr.update(value=res[0]), gr.update(value=res[1])

def build_interface():
    with gr.Blocks(title="f-lite Gradio GUI") as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate-btn",
                        scale=1,
                        visible=True
                    )
                    generate_forever_btn = gr.Button(
                        "Generate Automatically",
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
                with gr.Accordion("Advanced Prompt Options", open=False):
                    prompt_prefix = gr.Textbox(
                        label="Prompt Prefix",
                        lines=3,
                        value="",
                        elem_id="prompt-prefix-textbox",
                        info="Text to add before the final prompt"
                    )
                    prompt_suffix = gr.Textbox(
                        label="Prompt Suffix",
                        lines=3,
                        value="",
                        elem_id="prompt-suffix-textbox",
                        info="Text to add after the final prompt"
                    )
                with gr.Group():
                    with gr.Row():
                        enhance_prompt_checkbox = gr.Checkbox(
                            label="Enhance Prompt (SuperPrompt)",
                            value=True,
                            elem_id="enhance-prompt-checkbox",
                            scale=1,
                            info="Uses SuperPrompt to generate a more detailed prompt"
                        )
                        enhance_prompt_length = gr.Number(
                            label="Enhance Prompt Length",
                            value=256,
                            minimum=1,
                            maximum=512,
                            step=1,
                            precision=0,
                            elem_id="enhance-prompt-length",
                            scale=1,
                            info="Max number of tokens for enhanced prompt"
                        )
                    with gr.Row():
                        force_shuffle_enhance_prompt_checkbox = gr.Checkbox(
                            label="Force Shuffle Enhance Prompt",
                            value=False,
                            elem_id="force-shuffle-enhance-prompt-checkbox",
                            scale=1,
                            info="Enhance prompt randomly even if seed is locked"
                        )
                        force_shuffle_wildcards_checkbox = gr.Checkbox(
                            label="Force Shuffle Wildcards",
                            value=False,
                            elem_id="force-shuffle-wildcards-checkbox",
                            scale=1,
                            info="Randomize wildcards even if seed is locked"
                        )
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
                        label="Seed",
                        value=-1,
                        precision=0,
                        scale=1,
                        elem_id="seed-field"
                    )
                with gr.Group():
                    gr.Markdown("Augmented Parallel Guidance")
                    with gr.Row():
                        apg_enabled = gr.Checkbox(
                            label="AGP Enabled",
                            value=True,
                            elem_id="apg-enabled-checkbox",
                            scale=1,
                            info="Enables Augmented Parallel Guidance for more consistent results"
                        )
                        apg_orthogonal_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=0.2,
                            value=0.03,
                            step=0.001,
                            label="APG Orthogonal Threshold",
                            elem_id="apg-threshold-slider",
                            visible=False,
                            interactive=False,
                            scale=2,
                            info="The threshold for the orthogonal guidance"
                        )
                model = gr.Dropdown(
                    choices=["Freepik/F-Lite", "Freepik/F-Lite-Texture"],
                    label="Model",
                    value="Freepik/F-Lite",
                    elem_id="model-dropdown"
                )
            with gr.Column(scale=1):
                output = gr.Image(
                    label="Generated Image",
                    elem_id="output-image",
                    interactive=False,
                    show_download_button=False
                )
                status_area = gr.Textbox(
                    label="Status",
                    elem_id="status-area",
                    lines=20,
                    max_lines=40,
                    interactive=False
                )
                with gr.Row(elem_id="image-actions"):
                    open_folder_btn = gr.Button(
                        "📁 Open Output Folder",
                        elem_id="open-folder-btn",
                        scale=1
                    )
        
        generate_mode = gr.State("single")
        
        open_folder_btn.click(
            fn=open_output_folder,
            inputs=[],
            outputs=[]
        )
        
        cancel_btn.click(
            fn=set_cancel_flag,
            inputs=[],
            outputs=[cancel_btn],
        )
        
        apg_enabled.change(
            fn=lambda enabled_state: gr.update(visible=enabled_state, interactive=enabled_state),
            inputs=[apg_enabled],
            outputs=[apg_orthogonal_threshold]
        )
        
        generate_btn.click(
            lambda: "single",
            outputs=[generate_mode]
        ).then(
            generate,
            inputs=[prompt, negative_prompt, enhance_prompt_checkbox, force_shuffle_enhance_prompt_checkbox, force_shuffle_wildcards_checkbox, enhance_prompt_length, steps, guidance_scale, width, height, seed, model, generate_mode, status_area, output, apg_enabled, apg_orthogonal_threshold, prompt_prefix, prompt_suffix],
            outputs=[output, generate_btn, generate_forever_btn, cancel_btn, status_area],
        )
        
        generate_forever_btn.click(
            lambda: "forever",
            outputs=[generate_mode]
        ).then(
            generate,
            inputs=[prompt, negative_prompt, enhance_prompt_checkbox, force_shuffle_enhance_prompt_checkbox, force_shuffle_wildcards_checkbox, enhance_prompt_length, steps, guidance_scale, width, height, seed, model, generate_mode, status_area, output, apg_enabled, apg_orthogonal_threshold, prompt_prefix, prompt_suffix],
            outputs=[output, generate_btn, generate_forever_btn, cancel_btn, status_area],
        )
        
        preset_dropdown.change(
            fn=lambda x: next(((p["width"], p["height"]) for p in PRESET_RESOLUTIONS if p["name"] == x), (None, None)),
            inputs=[preset_dropdown],
            outputs=[width, height]
        )
        
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
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            let btn = document.querySelector('#generate-btn button') || document.querySelector('#generate-btn');
            if (btn && window.getComputedStyle(btn).display !== 'none') btn.click();
        }
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
