#!/usr/bin/env python
# coding=utf-8

import argparse
import datetime
import hashlib
import json
import os

import torch

# Import the pipeline and dataset class from your training script
from pipeline import FLitePipeline  # Adjust import as needed
from tqdm.auto import tqdm
from train import DiffusionDataset, encode_prompt_with_t5  # Reusing your dataset class


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VAE latents and text embeddings for DiT training")
    
    # Model parameters
    parser.add_argument("--pretrained_model_path", type=str, default=None, required=True,
                        help="Path to pretrained model")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset, supports CSV files or image directories")
    parser.add_argument("--base_image_dir", type=str, default=None,
                        help="Base directory for image paths in CSV files")
    parser.add_argument("--image_column", type=str, default="image_path",
                        help="Column name in CSV containing image paths")
    parser.add_argument("--caption_column", type=str, default="caption",
                        help="Column name in CSV containing text captions")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Image resolution for processing")
    parser.add_argument("--center_crop", action="store_true",
                        help="Whether to center crop images")
    parser.add_argument("--random_flip", action="store_true",
                        help="Whether to randomly flip images horizontally")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="precomputed_data",
                        help="Output directory for saving precomputed files")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for computation")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help="Mixed precision type")
    
    return parser.parse_args()

def get_file_hash(file_path):
    """Generate a short hash for a file path to use in filenames"""
    return hashlib.md5(file_path.encode()).hexdigest()[:10]

def get_text_hash(text):
    """Generate a short hash for a text to use in filenames"""
    return hashlib.md5(text.encode()).hexdigest()[:10]

def main():
    args = parse_args()
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    latents_dir = os.path.join(args.output_dir, "vae_latents")
    embeddings_dir = os.path.join(args.output_dir, "text_embeddings")
    os.makedirs(latents_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Load models
    print(f"Loading models from {args.pretrained_model_path}")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    pipeline = FLitePipeline.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=dtype
    )
    
    vae_model = pipeline.vae.to(device)
    text_encoder = pipeline.text_encoder.to(device)
    tokenizer = pipeline.tokenizer
    
    # Set models to eval mode
    vae_model.eval()
    text_encoder.eval()
    
    # Create dataset
    dataset = DiffusionDataset(
        data_path=args.data_path,
        base_image_dir=args.base_image_dir,
        image_column=args.image_column,
        caption_column=args.caption_column,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=False,  # Don't use random flip for precomputation
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Initialize mapping file
    mapping = {
        "metadata": {
            "date_created": datetime.datetime.now().isoformat(),
            "pretrained_model_path": args.pretrained_model_path,
            "data_path": args.data_path,
        },
        "entries": []
    }
    
    # Process all samples
    cached_text_embeddings = {}  # Cache to avoid recomputing the same captions
    
    print(f"Processing {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, (images_vae, metadatas) in enumerate(tqdm(data_loader)):
            # Normalize images to [-1, 1]
            images_vae = images_vae.to(device).to(dtype)
            images_vae.sub_(0.5).mul_(2.0)
            
            # Get captions
            captions = [metadata["long_caption"][0] for metadata in metadatas]
            
            # Encode images using VAE
            vae_latents = vae_model.encode(images_vae).latent_dist.sample()
            # Normalize latents
            vae_latents = (vae_latents - vae_model.config.shift_factor) * vae_model.config.scaling_factor
            
            # Process each sample in batch
            for i in range(len(captions)):
                # Get original data entry
                image_idx = batch_idx * args.batch_size + i
                if image_idx >= len(dataset):
                    break
                
                entry = dataset.data_entries[image_idx]
                image_path = entry["image_path"]
                caption = entry["caption"]
                
                # Generate file names using hashes
                image_hash = get_file_hash(image_path)
                text_hash = get_text_hash(caption)
                
                # Save VAE latent
                latent_filename = f"latent_{image_hash}.pt"
                latent_path = os.path.join(latents_dir, latent_filename)
                torch.save(vae_latents[i].cpu(), latent_path)
                
                # Create or retrieve text embedding
                if caption not in cached_text_embeddings:
                    text_embedding = encode_prompt_with_t5(
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        prompt=[caption],
                        device=device,
                        return_index=-8,  # Default index used in training
                    ).cpu()
                    
                    # Save text embedding
                    embedding_filename = f"embedding_{text_hash}.pt"
                    embedding_path = os.path.join(embeddings_dir, embedding_filename)
                    torch.save(text_embedding, embedding_path)
                    
                    cached_text_embeddings[caption] = embedding_filename
                else:
                    embedding_filename = cached_text_embeddings[caption]
                
                # Add entry to mapping
                mapping["entries"].append({
                    "image_path": image_path,
                    "caption": caption,
                    "latent_file": latent_filename,
                    "embedding_file": embedding_filename,
                    "image_hash": image_hash,
                    "text_hash": text_hash,
                })
                
            # Free memory
            del vae_latents
            torch.cuda.empty_cache()
    
    # Save mapping file
    mapping_path = os.path.join(args.output_dir, "precomputed_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Precomputation complete. Results saved to {args.output_dir}")
    print(f"Processed {len(mapping['entries'])} items")
    print(f"Unique text embeddings: {len(cached_text_embeddings)}")
    print(f"To use precomputed data, pass --use_precomputed_data --precomputed_data_dir {args.output_dir} to your training script")

if __name__ == "__main__":
    main()