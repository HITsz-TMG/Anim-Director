from diffusers import ( 
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionXLPipeline
)
import torch
import numpy as np
from PIL import Image
import os
import glob
import argparse
from tqdm import tqdm, trange
from utils.models import load_sam, load_owl


model_id = '/data/model/stable-diffusion-xl-base-1.0'
vae_id = '/data/model/sdxl-vae-fp16-fix'

prompt = "a photo of a {}, simple background, full body view, award winning photography, highly detailed"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry"
owl_threshold = 0.4

def main(args):
    num_images = args.num_images
    vae = vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
        )

    pipe = pipe.to("cuda")
    gen_prompt = prompt.format(args.gen_class)

    if args.out_dir is None:
        args.out_dir = f"dataset/prior/{args.gen_class.replace(' ', '_')}"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.gen_mask:
        predictor = load_sam("vit_h", 'cuda')
        processor, model = load_owl('cuda')
        if args.owl_query is None:
            args.owl_query = args.gen_class
        if args.mask_dir is None:
            args.mask_dir = args.out_dir
        os.makedirs(args.mask_dir, exist_ok=True)

    for i in trange(num_images):
        if args.gen_mask:
            for TRY in range(10):
                img = pipe(prompt=gen_prompt, negative_prompt=negative_prompt).images[0]
                inputs = processor(text=[args.owl_query], images=img, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    outputs = model(**inputs)

                target_sizes = torch.Tensor([img.size[::-1]])
                results = processor.post_process_object_detection(
                    outputs=outputs, 
                    target_sizes=target_sizes, 
                    threshold=owl_threshold,
                )
                boxes  = results[0]["boxes"]
                boxes = np.array(boxes.cpu().detach())

                if len(boxes) != 1:
                    print(f"not single object error")
                    continue
                else:
                    input_box = boxes[0]
                    predictor.set_image(np.array(img))
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    mask = Image.fromarray(masks[0].astype(np.uint8) * 255)
                    mask.save(f"{args.mask_dir}/mask_{i}.png")
                    break
                
                if TRY == 9:
                    raise ValueError("Failed to generate mask, try another class.")

        else:
            img = pipe(prompt=gen_prompt, negative_prompt=negative_prompt).images[0]
        img.save(f"{args.out_dir}/img_{i}.png")

            
            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval.")
    parser.add_argument("--gen_class", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=50)

    parser.add_argument("--gen_mask", action="store_true")
    parser.add_argument("--owl_query", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)


    args = parser.parse_known_args()[0]
    main(args)