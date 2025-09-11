import os
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from functools import partial
from IPython.display import clear_output

from diffusers import ( 
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionXLPipeline,
)
from diffusers.utils import make_image_grid
from archieve import MODEL_ROOT, SAVE_ROOT

from utils.sdxl_sdedit_pipeline import StableDiffusionXLImg2ImgPipeline

def load_sdxl_pipe(
    model_id=f'{MODEL_ROOT}/stable-diffusion-xl-base-1.0', 
    vae_id=f'{MODEL_ROOT}/sdxl-vae-fp16-fix', 
    refiner_id=f'{MODEL_ROOT}/stable-diffusion-xl-refiner-1.0',
    use_refiner=False,
    device='cuda:0'):

    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )

    if use_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            refiner_id,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to(device)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    clear_output()
    return pipe

def load_sdedit_pipe(
    model_id=f'{MODEL_ROOT}/stable-diffusion-xl-base-1.0', 
    vae_id=f'{MODEL_ROOT}/sdxl-vae-fp16-fix', 
    device='cuda:0', 
    **kwargs):

    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16
    )
    pipe.set_progress_bar_config(leave=False)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    clear_output()
    return pipe.to(device)

class gen_pipe:
    def __init__(self, 
        pipe=None,
        pipe_type='sdxl',
        model_id=f'{MODEL_ROOT}/stable-diffusion-xl-base-1.0', 
        vae_id=f'{MODEL_ROOT}/sdxl-vae-fp16-fix', 
        device='cuda:0',
        **kwargs
    ):
        super().__init__()
        if pipe:
            self.pipe = pipe
        else:
            load_pipe = {
                'sdxl': load_sdxl_pipe,
                'sdedit': load_sdedit_pipe,
            }
            self.pipe = load_pipe[pipe_type](
                model_id, 
                vae_id=vae_id, 
                device=device, 
                **kwargs
            )
        self.vae = self.pipe.vae
        self.image_processor = self.pipe.image_processor
        self.device = device
        self.img_list = []
        self.img_x0_list = []

    def sample(self, prompt, num_images=4, seed=0, make_grid=True, save_path=None, lora_scale=1.0, **kwargs):
        negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry"
        generator = torch.Generator(self.device).manual_seed(seed)

        imgs = []
        for i in trange(num_images, position=0, leave=False):
            img = self.pipe(prompt=prompt, 
                negative_prompt=negative_prompt,
                generator=generator,
                cross_attention_kwargs={'scale':lora_scale},
                **kwargs
            ).images[0]
            if save_path:
                img.save(os.path.join(save_path, f"{len(imgs):02d}.jpg"))
            imgs.append(img)
            clear_output(wait=True)
        clear_output(wait=True)
        self.img_list = imgs

        if make_grid:
            imgs_ = [img.resize((512,512)) for img in imgs]
            N = int(np.ceil(np.sqrt(num_images)))
            return make_image_grid(imgs_, N, N)
        else:
            return imgs
    
    def sample_with_x0_latents(self, prompt, save_x0_steps, num_images=4, seed=0, make_grid=True, save_path=None, lora_scale=1.0, **kwargs):
        negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry"
        generator = torch.Generator(self.device).manual_seed(seed)

        imgs = []
        imgs_x0 = []
        for i in trange(num_images, position=0, leave=False):
            img, img_x0 = self.pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                generator=generator,
                cross_attention_kwargs={'scale':lora_scale},
                return_dict=False,
                save_x0_steps=save_x0_steps,
                **kwargs
            )
            img, img_x0 = img[0], img_x0[0]
            # img_x0: [[(0, img), (1, img)]] 2d list of tuples
            if save_path:
                img.save(os.path.join(save_path, f"{len(imgs):02d}.jpg"))
                for j, img_ in img_x0:
                    img_.save(os.path.join(save_path, f"{len(imgs):02d}_x0_{i:02d}.jpg"))
            imgs.append(img)
            imgs_x0.append(img_x0)
            clear_output(wait=True)
        clear_output(wait=True)
        self.img_list = imgs
        self.img_x0_list = imgs_x0

        if make_grid:
            imgs_ = [img.resize((512,512)) for img in imgs]
            N = int(np.ceil(np.sqrt(num_images)))
            return make_image_grid(imgs_, N, N), imgs_x0
        else:
            return imgs, imgs_x0
    
