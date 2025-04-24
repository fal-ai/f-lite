import logging
from pathlib import Path
from typing import Optional, Union


import torch
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5TokenizerFast

from . import DiT, FLitePipeline

logger = logging.getLogger(__name__)


def load_f_lite_pt(
    model_path: Union[str, Path],
    device: torch.device,
    dtype: str = "float32",
    vae_path: Optional[Union[str, Path]] = None,
    text_encoder_path: Optional[Union[str, Path]] = None,
    lora_path: Optional[Union[str, Path]] = None,
    lora_scale: float = 1.0,
    lora_rank: int = 128,
    lora_target_modules: str = "qkv,q,context_kv,proj",
    # DiT model architecture parameters
    patch_size: int = 2,
    width: int = 3072,
    mlp_ratio: float = 4.0,
    cross_attn_input_size: int = 4096,
    residual_v: bool = True,
    train_bias_and_rms: bool = False,
    # Optimization options
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = False,
    compile_model: bool = False,
) -> FLitePipeline:
    """
    Load a F-Lite model from a PT file.

    Args:
        model_path: Path to the PT file containing the model weights
        device: Torch device to load the model on
        dtype: Data type to use for the model (float32, float16, bfloat16)
        vae_path: Path to the VAE model (defaults to "black-forest-labs/FLUX.1-schnell")
        text_encoder_path: Path to the text encoder model (defaults to "black-forest-labs/FLUX.1-schnell")
        lora_path: Path to LoRA weights (optional)
        lora_scale: LoRA scale factor
        lora_rank: LoRA rank
        lora_target_modules: Comma-separated list of target modules for LoRA
        patch_size: DiT patch size
        depth: DiT model depth
        width: DiT model width
        mlp_ratio: DiT MLP ratio
        cross_attn_input_size: DiT cross attention input size
        residual_v: Whether to use residual V in DiT
        train_bias_and_rms: Whether to train bias and RMS in DiT
        enable_vae_slicing: Whether to enable VAE slicing for memory efficiency
        enable_vae_tiling: Whether to enable VAE tiling
        compile_model: Whether to compile the model for faster inference

    Returns:
        FLitePipeline: The loaded F-Lite pipeline
    """
    # Convert string paths to Path objects
    model_path = Path(model_path)
    if vae_path:
        vae_path = Path(vae_path)
    if text_encoder_path:
        text_encoder_path = Path(text_encoder_path)
    if lora_path:
        lora_path = Path(lora_path)

    # Set torch dtype
    torch_dtype = getattr(torch, dtype)

    # Load DiT model
    logger.info(f"[F-Lite] Loading DiT model from {model_path}")
    state_dict = torch.load(str(model_path), map_location=device)
    
    # Infer depth from state dict by finding the maximum block index
    depth = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('blocks.')]) + 1
    logger.info(f"Inferred model depth from state dict: {depth}")

    # Initialize DiT model
    dit_model = DiT(
        in_channels=16,
        patch_size=patch_size,
        depth=depth,
        num_heads=width // 256,  # width/head_dim
        mlp_ratio=mlp_ratio,
        cross_attn_input_size=cross_attn_input_size,
        hidden_size=width,
        residual_v=residual_v,
        train_bias_and_rms=train_bias_and_rms,
    )

    # Clean up state dict keys
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v.clone().to(device, torch_dtype)
        for k, v in state_dict.items()
    }

    # Load state dict
    load_status = dit_model.load_state_dict(state_dict, strict=False, assign=True)
    logger.info(f"Loaded DiT checkpoint with status: {load_status}")

    # Add LoRA support if specified
    if lora_path is not None:
        try:
            from peft import LoraConfig, set_peft_model_state_dict

            logger.info(f"Loading LoRA weights from {lora_path}")
            # Parse target modules list
            target_modules = [module.strip() for module in lora_target_modules.split(",")]
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                target_modules=target_modules,
                bias="none",
                init_lora_weights="gaussian",
            )
            # Move model to device for adding LoRA adapters
            dit_model = dit_model.to(device, torch_dtype)
            # Add LoRA adapter
            dit_model.add_adapter(lora_config)
            # Load LoRA weights
            lora_state_dict = torch.load(str(lora_path), map_location=device)
            set_peft_model_state_dict(dit_model, lora_state_dict)
            logger.info(f"Successfully loaded LoRA weights with scale {lora_scale}")
        except ImportError:
            logger.error("Failed to import PEFT library. Make sure PEFT is installed.")
            raise
        except Exception as e:
            logger.error(f"Error loading LoRA weights: {e!s}")
            raise
    else:
        # If no LoRA weights, move model to device
        dit_model = dit_model.to(device, torch_dtype)

    # Load VAE
    logger.info("Loading VAE model")
    vae_path_str = str(vae_path) if vae_path else "black-forest-labs/FLUX.1-schnell"
    vae = AutoencoderKL.from_pretrained(vae_path_str, torch_dtype=torch.float32, subfolder="vae").to(device)
    vae.to(memory_format=torch.channels_last)

    # Load text encoder & tokenizer
    logger.info("Loading text encoder and tokenizer")
    text_encoder_path_str = str(text_encoder_path) if text_encoder_path else "black-forest-labs/FLUX.1-schnell"
    tokenizer = T5TokenizerFast.from_pretrained(text_encoder_path_str, subfolder="tokenizer_2")
    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_path_str,
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
    ).to(device)

    # Create pipeline
    pipe = FLitePipeline(dit_model=dit_model, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer)

    # Apply optimizations
    if enable_vae_slicing:
        pipe.enable_vae_slicing()
    if enable_vae_tiling:
        pipe.enable_vae_tiling()

    # Compile model if requested
    if compile_model:
        logger.info("Compiling models...")
        pipe.dit_model = torch.compile(pipe.dit_model, mode="reduce-overhead", fullgraph=True)
        if hasattr(pipe.vae, "encode"):
            pipe.vae.encode = torch.compile(pipe.vae.encode, mode="reduce-overhead")
        forward_func = pipe.text_encoder.forward
        pipe.text_encoder.forward = torch.compile(forward_func, mode="reduce-overhead")
        logger.info("Warming up compiled models...")
        # Dummy inference to warm up the compiled models
        prompt = "a photo of a cat holding a sign that says hello world"
        _ = pipe(prompt=prompt, generator=torch.manual_seed(1))

    return pipe 