import torch
from diffusers import FluxPipeline

class FluxGenerator:
    def __init__(self, model_path="Tools/personalize-anything/checkpoints/FLUX1-dev", device="cuda", torch_dtype=torch.bfloat16, cpu_offload=True):
        self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
    
    def generate(self, prompt, height=768, width=1356, guidance_scale=3.5, 
                num_inference_steps=50, max_sequence_length=512, seed=0):

        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]
        return image
    
    def save_image(self, image, savepath):
        """Save the generated image to a file"""
        image.save(savepath)


# # Example usage:
# if __name__ == "__main__":
#     generator = FluxGenerator()
#     prompt = "Background image, Medium shot: Vast sand dunes, sunrise hues, cracked earth, sparse vegetation, and a gradient sky. Minimalism style, 2D animation."
#     image = generator.generate(prompt)
#     generator.save_image(image, savepath="Tools/Flux/image.jpg")
