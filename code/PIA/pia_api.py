import os
import numpy as np
import torch
from omegaconf import OmegaConf

from PIA.animatediff.pipelines import I2VPipeline
from PIA.animatediff.utils.util import preprocess_img, save_videos_grid


def seed_everything(seed):
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


class PiaAPI:
    def __init__(self, prompts, input_path, input_name, save_path):
        self.config = OmegaConf.create({
            'base': 'code/PIA/example/config/base.yaml',
            'prompts': prompts,
            'n_prompt': ['wrong white balance, dark, sketches, worst quality, low quality, deformed, distorted, disfigured, bad eyes, wrong lips, weird mouth, bad teeth, mutated hands and fingers, bad anatomy, wrong anatomy, amputation, extra limb, missing limb, floating,limbs, disconnected limbs, mutation, ugly, disgusting, bad_pictures, negative_hand-neg'],
            'validation_data': {
                'input_name': input_name,
                'validation_input_path': input_path,
                'save_path': save_path,
                'mask_sim_range': [0]
            },
            'generate': {
                'use_lora': False,
                'use_db': True,
                'global_seed': 10201403011320481249,
                'lora_path': "",
                'db_path': "code/PIA/models/DreamBooth_LoRA/rcnzCartoon3d_v10.safetensors",
                'lora_alpha': 0.5,
                'video_length': 16
            }
        })
        
        self.config = OmegaConf.merge(OmegaConf.load(self.config.base), self.config)


    def generate(self):
        config = self.config

        os.makedirs(config.validation_data.save_path, exist_ok=True)
        folder_num = len(os.listdir(config.validation_data.save_path))
        # target_dir = f"{config.validation_data.save_path}/{folder_num}/"
        target_dir = f"{config.validation_data.save_path}"

        # Prepare paths and pipeline
        base_model_path = config.pretrained_model_path
        unet_path = config.generate.model_path
        dreambooth_path = config.generate.db_path
        lora_path = config.generate.get("lora_path", None)
        lora_alpha = config.generate.get("lora_alpha", 0)

        validation_pipeline = I2VPipeline.build_pipeline(
            config,
            base_model_path,
            unet_path,
            dreambooth_path,
            lora_path,
            lora_alpha,
        )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(config.generate.global_seed)

        global_inf_num = 0

        os.makedirs(target_dir, exist_ok=True)

        print(f"using unet      : {unet_path}")
        print(f"using DreamBooth: {dreambooth_path}")
        print(f"using Lora      : {lora_path}")

        sim_ranges = config.validation_data.mask_sim_range
        if isinstance(sim_ranges, int):
            sim_ranges = [sim_ranges]

        OmegaConf.save(config, os.path.join(target_dir, "config.yaml"))
        generator.manual_seed(config.generate.global_seed)
        seed_everything(config.generate.global_seed)

        # Load image
        img_root = config.validation_data.validation_input_path
        input_name = config.validation_data.input_name
        image_name = os.path.join(img_root, f"{input_name}.jpg")
        if not os.path.exists(image_name):
            image_name = os.path.join(img_root, f"{input_name}.png")
            if not os.path.exists(image_name):
                raise ValueError("image_name should be .jpg or .png")

        image, gen_height, gen_width = preprocess_img(image_name)
        config.generate.sample_height = gen_height
        config.generate.sample_width = gen_width

        for sim_range in sim_ranges:
            print(f"using sim_range : {sim_range}")
            config.validation_data.mask_sim_range = sim_range
            prompt_num = 0
            for prompt, n_prompt in zip(config.prompts, config.n_prompt):
                print(f"using n_prompt  : {n_prompt}")
                prompt_num += 1
                for single_prompt in prompt:
                    print(f" >>> Begin test {global_inf_num} >>>")
                    global_inf_num += 1
                    sample = validation_pipeline(
                        image=image,
                        prompt=single_prompt,
                        generator=generator,
                        video_length=config.generate.video_length,
                        height=config.generate.sample_height,
                        width=config.generate.sample_width,
                        negative_prompt=n_prompt,
                        mask_sim_template_idx=config.validation_data.mask_sim_range,
                        **config.validation_data,
                    ).videos
                    print(target_dir + ".gif")
                    save_videos_grid(sample, target_dir + ".gif")
                    print(f" <<< test {global_inf_num} Done <<<")
        print(" <<< Test Done <<<")
