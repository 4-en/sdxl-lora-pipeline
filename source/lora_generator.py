from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import os

import PIL

# simple class to generate images while using lora
class LoRAGenerator:
    """A class to generate images using a trained LoRA model.
    Use text2img() to generate images from text and img2img() to generate images from images.
    """

    def __init__(self, model: str = "stabilityai/stable-diffusion-xl-base-1.0", lora_path: str | None = None, device: str = "cuda"):
        self.text2img = StableDiffusionXLPipeline.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        if device:
            self.text2img = self.text2img.to(device)

        self.lora_loaded = False

        if lora_path:
            self.load_lora(lora_path)

        self.img2img = StableDiffusionXLImg2ImgPipeline(**self.text2img.components)

    
    def load_lora(self, lora_path: str):
        # check if path is dir
        if os.path.isdir(lora_path):
            lora_def = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
            if not os.path.exists(lora_def):
                # check for .saftensors, .pt, .pth, .ckpt, .bin
                extensions = [".safetensors", ".pt", ".pth", ".ckpt", ".bin"]
                all_files = os.listdir(lora_path)
                found = []
                for ext in extensions:
                    found += [f for f in all_files if f.endswith(ext)]

                if len(found) == 0:
                    print(f"No LoRA weights found in {lora_path}")
                    return
                
                if len(found) > 1:
                    print(f"Multiple LoRA weights found in {lora_path}: {found}")
                    return
                
                lora_path = os.path.join(lora_path, found[0])


        if not os.path.exists(lora_path):
            print(f"LoRA weights not found at {lora_path}")
            return

        self.text2img.load_lora_weights(lora_path)
        self.lora_loaded = True

    def unload_lora(self):
        self.text2img.unload_lora_weights()
        self.lora_loaded = False

    def load_image(self, img_path: str) -> PIL.Image:
        return PIL.Image.open(img_path)