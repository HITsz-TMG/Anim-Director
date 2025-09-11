import torch
from diffusers import StableDiffusion3Pipeline

class StableDiffusionAPI:
    def __init__(self, model_name="stabilityai/stable-diffusion-3-medium-diffusers", device="cuda"):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe.to(device)
        self.device = device

    def generate_image(self, path, prompt, negative_prompt="", num_inference_steps=28, height=1024, width=1024, guidance_scale=7.0):
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        ).images[0]
        image.save(path)

# generator = StableDiffusionAPI()
# generator.generate_image(path='', prompt="a photo of a cat holding a sign that says hello world")
