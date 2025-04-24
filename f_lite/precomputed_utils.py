import os
import json
import time
import math
from einops import rearrange
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.ImageOps import exif_transpose

class PrecomputedDiffusionDataset(Dataset):
    """
    Dataset for loading precomputed VAE latents and text embeddings for DiT fine-tuning.
    """
    def __init__(
        self,
        precomputed_data_dir,
        data_path=None,  # Original data path (only used for logging)
        random_flip=False,
    ):
        self.precomputed_data_dir = precomputed_data_dir
        self.random_flip = random_flip
        
        # Load mapping file
        mapping_path = os.path.join(precomputed_data_dir, "precomputed_mapping.json")
        if not os.path.exists(mapping_path):
            raise ValueError(f"Mapping file not found at {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)
        
        self.data_entries = self.mapping["entries"]
        print(f"Loaded precomputed dataset with {len(self.data_entries)} entries")
        
        # Paths to directories
        self.latents_dir = os.path.join(precomputed_data_dir, "vae_latents")
        self.embeddings_dir = os.path.join(precomputed_data_dir, "text_embeddings")
        
        # Verify directories exist
        if not os.path.exists(self.latents_dir):
            raise ValueError(f"VAE latents directory not found at {self.latents_dir}")
        if not os.path.exists(self.embeddings_dir):
            raise ValueError(f"Text embeddings directory not found at {self.embeddings_dir}")
    
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, index):
        entry = self.data_entries[index]
        
        # Load precomputed VAE latent
        latent_path = os.path.join(self.latents_dir, entry["latent_file"])
        vae_latent = torch.load(latent_path)
        
        # Load precomputed text embedding
        embedding_path = os.path.join(self.embeddings_dir, entry["embedding_file"])
        text_embedding = torch.load(embedding_path)
        
        # Apply random flip if specified (needs to be applied to latent)
        if self.random_flip and torch.rand(1).item() < 0.5:
            vae_latent = torch.flip(vae_latent, dims=[2])  # Flip width dimension
        
        # Return latent, text embedding, and metadata
        return (
            vae_latent,  # VAE latent instead of image tensor
            text_embedding,  # Text embedding instead of text
            [{
                "long_caption": entry["caption"],  # Still include the caption for logging/debugging
            }]
        )

class PrecomputedResolutionBucketSampler(torch.utils.data.BatchSampler):
    """Group latents by resolution to ensure consistent resolution within a batch"""
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group latents by resolution
        self.buckets = {}
        for idx in range(len(dataset)):
            # Load latent to get its shape
            entry = dataset.data_entries[idx]
            latent_path = os.path.join(dataset.latents_dir, entry["latent_file"])
            latent = torch.load(latent_path)
            resolution = (latent.shape[2], latent.shape[3])  # (height, width)
            
            if resolution not in self.buckets:
                self.buckets[resolution] = []
            self.buckets[resolution].append(idx)
        
        print(f"Created {len(self.buckets)} resolution buckets for precomputed data")
    
    def __iter__(self):
        # Create batches within each resolution bucket
        batches = []
        for resolution, indices in self.buckets.items():
            # Shuffle indices within buckets if requested
            if self.shuffle:
                import random
                indices = random.sample(indices, len(indices))
            
            # Create complete batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # Shuffle batch order if requested
        if self.shuffle:
            import random
            random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self):
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.buckets.values())
        else:
            return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.buckets.values())

def create_precomputed_data_loader(
    precomputed_data_dir,
    batch_size,
    shuffle=True,
    num_workers=4,
    random_flip=False,
    use_resolution_buckets=True,
):
    """Create a data loader for precomputed data"""
    # Create dataset
    dataset = PrecomputedDiffusionDataset(
        precomputed_data_dir=precomputed_data_dir,
        random_flip=random_flip,
    )
    
    # Create sampler - either resolution bucket or standard
    if use_resolution_buckets:
        sampler = PrecomputedResolutionBucketSampler(dataset, batch_size, shuffle=shuffle)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Standard approach
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    return data_loader

def forward_with_precomputed_data(
    dit_model,
    batch,
    device,
    global_step,
    master_process,
    generator=None,
    binnings=None,
    batch_multiplicity=None,
    bs_rampup=None,
    batch_size=None,
):
    """
    Forward pass for DiT model using precomputed VAE latents and text embeddings.
    
    Args:
        dit_model: DiT model
        batch: Batch of data (vae_latents, text_embeddings, metadata)
        device: Target device
        global_step: Current training step
        master_process: Whether this is the master process
        generator: Random number generator
        binnings: Tuple of (binning_dict, binning_count) for timestep binning
        batch_multiplicity: Factor to repeat batch
        bs_rampup: Steps for batch size rampup
        batch_size: Target batch size
    """
    (vae_latent, caption_encoded, metadatas) = batch
    
    # Process latents and embeddings
    preprocess_start = time.time()
    vae_latent = vae_latent.to(device).to(torch.bfloat16)
    caption_encoded = caption_encoded.to(device).to(torch.bfloat16)

    if caption_encoded.dim() == 4:
        caption_encoded = caption_encoded.squeeze(1)
     
    # Apply batch multiplicity if needed
    if batch_multiplicity is not None:
        # Repeat the batch by factor of batch_multiplicity
        vae_latent = vae_latent.repeat(batch_multiplicity, 1, 1, 1)
        caption_encoded = caption_encoded.repeat(batch_multiplicity, 1, 1)
        
        # Randomly zero out some caption_encoded (simulates classifier-free guidance)
        do_zero_out = torch.rand(caption_encoded.shape[0], device=device) < 0.01
        caption_encoded[do_zero_out] = 0
    
    # Batch size rampup - gradually increase batch size during training
    if bs_rampup is not None and global_step < bs_rampup:
        target_bs = math.ceil((global_step + 1) * batch_size / bs_rampup / 4) * 4  # Round to multiple of 4
        if vae_latent.size(0) > target_bs:
            keep_indices = torch.randperm(vae_latent.size(0))[:target_bs]
            vae_latent = vae_latent[keep_indices]
            caption_encoded = caption_encoded[keep_indices]
    
    batch_size = vae_latent.size(0)
    
    # TIMEStep Sampling for diffusion process
    image_token_size = vae_latent.shape[2] * vae_latent.shape[3]  # h * w
    z = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    alpha = 2 * math.sqrt(image_token_size / (64 * 64))
    
    # Mix uniform and lognormal sampling for better coverage of timesteps
    do_uniform = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator) < 0.1
    uniform = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
    t = torch.nn.Sigmoid()(z)
    lognormal = t * alpha / (1 + (alpha - 1) * t)
    
    # Final timesteps
    t = torch.where(~do_uniform, lognormal, uniform).to(torch.bfloat16)
    
    # Generate noise
    noise = torch.randn(
        vae_latent.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    
    preprocess_time = time.time() - preprocess_start
    if master_process:
        print(f"Preprocessing took {preprocess_time*1000:.2f}ms, alpha={alpha:.2f}")
    
    # Forward pass through DiT
    forward_start = time.time()
    
    # Create noisy latents
    tr = t.reshape(batch_size, 1, 1, 1)
    z_t = vae_latent * (1 - tr) + noise * tr
    
    # Velocity prediction objective
    v_objective = vae_latent - noise
     
    # Forward through model
    output = dit_model(z_t, caption_encoded, t)
    
    targ = rearrange(v_objective, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2)
    pred = rearrange(output, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2)

    diffusion_loss_batchwise = (
        (targ.float() - pred.float()).pow(2).mean(dim=(1, 2))
    )
    diffusion_loss = diffusion_loss_batchwise.mean()  
    
    # Combine losses
    total_loss = diffusion_loss
    
    # Record timestep binning statistics
    tbins = [min(int(_t * 10), 9) for _t in t]
    if binnings is not None:
        (
            diffusion_loss_binning,
            diffusion_loss_binning_count,
        ) = binnings
        for idx, tb in enumerate(tbins):
            diffusion_loss_binning[tb] += diffusion_loss_batchwise[idx].item()
            diffusion_loss_binning_count[tb] += 1
    
    forward_time = time.time() - forward_start
    if master_process:
        print(f"Forward pass took {forward_time*1000:.2f}ms")
    
    return total_loss, diffusion_loss