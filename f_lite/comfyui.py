import torch
import numpy as np
from PIL import Image
import os
import comfy.model_management as mm

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def check_and_download_model(model_path, repo_id):
    import folder_paths
    model_path = os.path.join(folder_paths.models_dir, "F-Lite", model_path)

    if not os.path.exists(model_path):
        print(f"Downloading {repo_id} model to {model_path} ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt", ".git", ".gitattributes"])
    return model_path

class FLiteNode:

    def __init__(self):
        self.NODE_NAME = 'F-Lite image generator'
        self.model_name = ""
        self.pipe = None
        self.compile_model = None
        self.cpu_offload = None


    @classmethod
    def INPUT_TYPES(self):
        model_list =['F-Lite', 'F-Lite-Texture']
        default_prompt = "A photorealistic 3D render of a charming, mischievous young boy, approximately eight years old, possessing the endearingly unusual features of long, floppy donkey ears that droop playfully over his shoulders and a surprisingly small, pink pig nose that twitches slightly.  His eyes, a sparkling, intelligent hazel, are wide with a hint of playful mischief, framed by slightly unruly, sandy-brown hair that falls in tousled waves across his forehead.  He's dressed in a simple, slightly oversized, worn denim shirt and patched-up corduroy trousers, hinting at a life spent playing outdoors. The lighting is soft and natural, casting gentle shadows that highlight the texture of his skin – slightly freckled and sun-kissed, suggesting time spent in the sun.  His expression is one of curious anticipation, his lips slightly parted as if he's about to speak or perhaps is listening intently. The background is a subtly blurred pastoral scene, perhaps a sun-dappled meadow with wildflowers, enhancing the overall whimsical and slightly surreal nature of the character.  The overall style aims for a blend of realistic rendering with a touch of whimsical cartoonishness, capturing the unique juxtaposition of the boy's human features and his animalistic ears and nose."
        default_negative_prompt = ""
        return {
            "required": {
                "model": (model_list,),
                "prompt":("STRING", {"default":default_prompt, "multiline": True}),
                "negative_prompt":("STRING", {"default":default_negative_prompt, "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1e14}),
                "guidance_scale": ("FLOAT", {"default": 6, "min": 0.1, "max": 100, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "width": ("INT", {"default": 1344, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 896, "min": 16, "max": 4096, "step": 16}),
                "compile_model": ("BOOLEAN", {"default": False,}),
                "cpu_offload": ("BOOLEAN", {"default": True,}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'f_lite'
    CATEGORY = 'F-Lite'

    def f_lite(self, model, prompt, negative_prompt, seed, guidance_scale, batch_size, steps, width, height, compile_model, cpu_offload):

        ret_images = []

        from .pipeline import FLitePipeline

        if self.model_name != model or self.compile_model != compile_model or self.cpu_offload != cpu_offload:
            model_path = check_and_download_model(model, f"Freepik/{model}")
            self.model_name = model

            self.compile_model = compile_model
            self.cpu_offload = cpu_offload
            self.pipe = FLitePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            
            if self.compile_model:
                self.pipe.vae = torch.compile(self.pipe.vae)
                self.pipe.dit_model = torch.compile(self.pipe.dit_model)
                self.pipe.text_encoder = torch.compile(self.pipe.text_encoder)

            if cpu_offload:
                # Memory usage reduction
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(mm.get_torch_device())

            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()


        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=torch.Generator().manual_seed(seed),
        ).images
        for i in image:
            ret_images.append(pil2tensor(i))

        return (torch.cat(ret_images, dim=0),)



NODE_CLASS_MAPPINGS = {
    "F-Lite": FLiteNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "F-Lite": "F-Lite AI Image Generator"
}

NODE_DESCRIPTION_MAPPINGS = {
    "F-Lite": "F-Lite AI Image Generator"
} 