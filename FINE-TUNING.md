# Diffusion Transformer (DiT) Fine-tuning Tool

A powerful tool for fine-tuning Text-to-Image Diffusion Transformer (DiT) models with custom datasets. This tool supports both full fine-tuning and memory-efficient LoRA fine-tuning with your own images and captions.

## Table of Contents
- [Data Preparation](#data-preparation)
- [Training Parameters](#training-parameters)
- [Basic Usage](#basic-usage)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [Advanced Usage](#advanced-usage)
- [Sampling & Evaluation](#sampling--evaluation)
- [Memory Optimization with Precomputation](#memory-optimization-with-precomputation)
- [Troubleshooting](#troubleshooting)

## Data Preparation

### CSV Format
The recommended approach is to organize your training data in CSV files with the following format:

```
image_path,caption
images/dog001.jpg,a cute brown dog sitting on the grass
images/cat042.jpg,an orange cat sleeping on a windowsill
landscape/mountain105.jpg,a beautiful mountain landscape at sunrise
portraits/woman023.jpg,portrait of a woman with long blonde hair
```

**Key Columns:**
- `image_path`: Path to the image file (relative to the `base_image_dir`)
- `caption`: Text description of the image (used to train the model)

> Note: You can customize the column names using the `--image_column` and `--caption_column` parameters.

### Image Directory Structure
The `base_image_dir` is the root folder containing all your training images. Paths in your CSV should be relative to this directory.

For example, if your `base_image_dir` is `/data/training_images/`, then the full paths would be:
```
/data/training_images/images/dog001.jpg
/data/training_images/images/cat042.jpg
/data/training_images/landscape/mountain105.jpg
/data/training_images/portraits/woman023.jpg
```

You can organize your images in subdirectories however you prefer - the script will find them as long as the paths in the CSV are correct.

### Directory-Based Training (No CSV)
If you don't want to create a CSV file, you can also just point the script to a directory of images:

```bash
python -m f_lite.train --train_data_path /path/to/image_directory --other_params ...
```

In this case:
- All images in the directory will be used for training
- The filenames (without extension) will be used as captions
  - For example, `beautiful_sunset.jpg` becomes the caption "beautiful sunset"

This approach is simpler but gives you less control over the captions associated with each image.

## Training Parameters

### Core Parameters
```
--pretrained_model_path    Path to pretrained model
--train_data_path          Training data CSV file or image directory
--val_data_path            Validation data CSV file or image directory (optional)
--base_image_dir           Base directory for images
--output_dir               Output directory for saving model and checkpoints
```

### Data Parameters
```
--image_column             Column name in CSV containing image paths (default: "image_path")
--caption_column           Column name in CSV containing captions (default: "caption")
--resolution               Resolution for training images (optional, native resolution used if not specified)
--center_crop              Enable center cropping
--random_flip              Enable random horizontal flipping
--use_resolution_buckets   Group images with same resolution into batches
```

### Training Control Parameters
```
--train_batch_size         Batch size for training (default: 1)
--eval_batch_size          Batch size for evaluation (default: 1)
--num_epochs               Number of training epochs (default: 1)
--max_steps                Maximum number of training steps (overrides epochs if specified)
--gradient_accumulation_steps  Number of gradient accumulation steps (default: 1)
--learning_rate            Learning rate (default: 1e-4)
--weight_decay             Weight decay coefficient (default: 0.01)
--lr_scheduler             Learning rate scheduler type ['linear', 'cosine', 'constant'] (default: linear)
--num_warmup_steps         Number of warmup steps (default: 0)
--use_8bit_adam            Use 8-bit Adam optimizer to save memory
```

### LoRA Parameters
```
--use_lora                 Enable LoRA for memory-efficient fine-tuning
--train_only_lora          Freeze base model and train only LoRA weights
--lora_rank                Rank of LoRA matrices (default: 64)
--lora_alpha               Scaling factor for LoRA (default: 64)
--lora_dropout             Dropout probability for LoRA layers (default: 0.0)
--lora_target_modules      Comma-separated list of target modules for LoRA (default: "qkv,q,context_kv,proj")
--lora_checkpoint          Path to LoRA checkpoint to resume from
```

### Sampling & Evaluation Parameters
```
--sample_every             Generate sample images every X steps (default: 500)
--eval_every               Run evaluation every X steps (default: 500)
--sample_prompts_file      Path to a text file containing prompts for sample image generation
--batch_multiplicity       Repeat batch samples this many times (default: 1)
```

### Advanced Options
```
--gradient_checkpointing   Enable gradient checkpointing to save memory
--mixed_precision          Mixed precision training ['no', 'fp16', 'bf16']
--max_grad_norm            Maximum gradient norm for gradient clipping (default: 1.0)
--seed                     Random seed for reproducibility
--report_to                Logging integration to use ['tensorboard', 'wandb', 'all'] (default: tensorboard)
--project_name             Project name for wandb logging (default: "dit-finetune")
```

## Basic Usage

Basic training command:

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --train_data_path ./data/train_data.csv \
  --base_image_dir ./data/images \
  --output_dir ./dit_finetuned \
  --train_batch_size 4 \
  --num_epochs 5 \
  --learning_rate 1e-5
```

## LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is a memory-efficient fine-tuning technique that significantly reduces VRAM requirements while still achieving excellent results. This is highly recommended for consumer GPUs:

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --train_data_path ./data/train_data.csv \
  --base_image_dir ./data/images \
  --output_dir ./dit_finetuned \
  --use_lora \
  --train_only_lora \
  --lora_rank 16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_epochs 20
```

## Advanced Usage

### Using Resolution Buckets for Mixed-Resolution Datasets

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --train_data_path ./data/train_data.csv \
  --base_image_dir ./data/images \
  --output_dir ./dit_finetuned \
  --use_resolution_buckets \
  --train_batch_size 4
```

### Enable Gradient Checkpointing to Save Memory

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --train_data_path ./data/train_data.csv \
  --base_image_dir ./data/images \
  --output_dir ./dit_finetuned \
  --gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --train_batch_size 2
```

### Using 8-bit Optimizer to Further Save Memory

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --train_data_path ./data/train_data.csv \
  --base_image_dir ./data/images \
  --output_dir ./dit_finetuned \
  --use_8bit_adam \
  --train_batch_size 8
```

## Sampling & Evaluation

Generate samples during training to monitor progress:

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --train_data_path ./data/train_data.csv \
  --base_image_dir ./data/images \
  --output_dir ./dit_finetuned \
  --sample_every 100 \
  --sample_prompts_file ./prompts.txt
```

Where `prompts.txt` contains one prompt per line:

```
a beautiful landscape with mountains
a cute cat sitting on a windowsill
a cyberpunk cityscape at night
a portrait of a woman with flowers in her hair
```

## Memory Optimization with Precomputation

For extremely memory-constrained scenarios, you can use precomputation to significantly reduce VRAM usage. This approach works by:
1. Precomputing and storing VAE latents for all training images
2. Precomputing and storing text embeddings for all captions
3. Training without loading the VAE and text encoder models into memory

### Precomputation Setup

First, run the precomputation script to process your dataset:

```bash
python -m f_lite.precompute_embeddings \
  --pretrained_model_path ./pretrained_models/dit_model \
  --data_path ./data/train_data.csv \
  --base_image_dir ./data/images \
  --output_dir ./precomputed_data \
  --resolution 512 \
  --mixed_precision bf16
```

### Training with Precomputed Data

Once precomputation is complete, train using the precomputed data:

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --use_precomputed_data \
  --precomputed_data_dir ./precomputed_data \
  --output_dir ./dit_finetuned \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --gradient_checkpointing
```

### Combining with LoRA for Maximum Efficiency

For the most memory-efficient training setup, combine precomputation with LoRA:

```bash
python -m f_lite.train \
  --pretrained_model_path ./pretrained_models/dit_model \
  --use_precomputed_data \
  --precomputed_data_dir ./precomputed_data \
  --output_dir ./dit_finetuned_lora \
  --use_lora \
  --train_only_lora \
  --lora_rank 16 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --gradient_checkpointing
```

### Benefits of Precomputation

- **Maximum Memory Savings**: Removes large VAE and text encoder models from GPU memory
- **Faster Training**: Eliminates redundant encoding operations for each batch
- **No Quality Loss**: Results are identical to standard training
- **Low Disk Overhead**: Precomputed files typically use less storage than original images

## Troubleshooting

### Common Errors

1. **CUDA Out of Memory**
   - Reduce `train_batch_size`
   - Enable `--gradient_checkpointing`
   - Increase `--gradient_accumulation_steps`
   - Use `--use_8bit_adam`
   - Enable LoRA with `--use_lora --train_only_lora`

2. **Resolution Mismatch Errors**
   - Use `--use_resolution_buckets` to handle images with different resolutions

3. **Image Files Not Found**
   - Ensure `--base_image_dir` is correctly set
   - Check that paths in CSV are relative to base_image_dir

4. **VAE Encoding Errors**
   - Ensure input images are in correct format (RGB mode)

### Advanced Tips
- For large datasets, use smaller learning rates (1e-5 to 5e-6)
- For small datasets, enable stronger regularization (increase weight_decay)
- Start with shorter training (1-2 epochs) and extend as needed
- When using LoRA, lower ranks (4-16) work well for small datasets, higher ranks (32-64) for larger datasets
- Use the sample_prompts_file to monitor specific concepts during training

We hope this guide helps you successfully fine-tune your DiT model! Feel free to reach out with any questions.
