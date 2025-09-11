# Open Source Model Licensed under the Apache License Version 2.0 
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited 
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.l

import os
import warnings
import argparse
import time
from PIL import Image
import torch
from glob import glob

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from third_party.mesh_baker import MeshBaker
from third_party.check import check_bake_available

try:
    from third_party.mesh_baker import MeshBaker
    assert check_bake_available()
    BAKE_AVAILEBLE = True
except Exception as err:
    print(err)
    print("import baking related fail, run without baking")
    BAKE_AVAILEBLE = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_lite", default=False, action="store_true"
    )
    parser.add_argument(
        "--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str
    )
    parser.add_argument(
        "--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str
    )
    parser.add_argument(
        "--text2image_path", default="weights/hunyuanDiT", type=str
    )
    parser.add_argument(
        "--save_folder", default="./outputs/test/", type=str
    )
    parser.add_argument(
        "--text_prompt", default="", type=str,
    )
    parser.add_argument(
        "--image_prompt", default="", type=str
    )
    parser.add_argument(
        "--device", default="cuda:0", type=str
    )
    parser.add_argument(
        "--t2i_seed", default=0, type=int
    )
    parser.add_argument(
        "--t2i_steps", default=25, type=int
    )
    parser.add_argument(
        "--gen_seed", default=0, type=int
    )
    parser.add_argument(
        "--gen_steps", default=50, type=int
    )
    parser.add_argument(
        "--max_faces_num", default=120000, type=int, 
        help="max num of face, suggest 120000 for vertex color, 10000 for texture/baking color"
    )
    parser.add_argument(
        "--save_memory", default=False, action="store_true"
    )
    parser.add_argument(
        "--do_texture_mapping", default=False, action="store_true"
    )
    parser.add_argument(
        "--do_render", default=False, action="store_true"
    )
    parser.add_argument(
        "--do_bake", default=False, action="store_true"
    )
    parser.add_argument(
        "--bake_align_times", default=3, type=int,
        help="align times between view image and mesh, suggest 1~6"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    assert not (args.text_prompt and args.image_prompt), "Text and image can only be given to one"
    assert args.text_prompt or args.image_prompt,        "Text and image can only be given to one"

    # init model
    st = time.time()
    rembg_model = Removebg()
    image_to_views_model = Image2Views(
        device=args.device, 
        use_lite=args.use_lite,
        save_memory=args.save_memory
    )
    
    views_to_mesh_model = Views2Mesh(
        args.mv23d_cfg_path, 
        args.mv23d_ckt_path, 
        args.device, 
        use_lite=args.use_lite,
        save_memory=args.save_memory
    )
    
    if args.text_prompt:
        text_to_image_model = Text2Image(
            pretrain = args.text2image_path,
            device = args.device, 
            save_memory = args.save_memory
        )
        
    if args.do_bake and BAKE_AVAILEBLE:
        mesh_baker = MeshBaker(
            device = args.device,
            align_times = args.bake_align_times
        )
            
    if check_bake_available():
        gif_renderer = GifRenderer(device=args.device)
        
    print(f"Init Models cost {time.time()-st}s")
    
    # ---- ----- ---- ---- ---- ----

    os.makedirs(args.save_folder, exist_ok=True)

    # stage 1, text to image
    if args.text_prompt:
        res_rgb_pil = text_to_image_model(
            args.text_prompt, 
            seed=args.t2i_seed,  
            steps=args.t2i_steps
        )
        res_rgb_pil.save(os.path.join(args.save_folder, "img.jpg"))
    elif args.image_prompt:
        res_rgb_pil = Image.open(args.image_prompt)

    # stage 2, remove back ground
    res_rgba_pil = rembg_model(res_rgb_pil)
    res_rgba_pil.save(os.path.join(args.save_folder, "img_nobg.png"))

    # stage 3, image to views
    (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
        res_rgba_pil,
        seed = args.gen_seed,
        steps = args.gen_steps
    )
    views_grid_pil.save(os.path.join(args.save_folder, "views.jpg"))

    # stage 4, views to mesh
    views_to_mesh_model(
        views_grid_pil, 
        cond_img, 
        seed = args.gen_seed,
        target_face_count = args.max_faces_num,
        save_folder = args.save_folder,
        do_texture_mapping = args.do_texture_mapping
    )
    
    # stage 5, baking
    mesh_file_for_render = None
    if args.do_bake and BAKE_AVAILEBLE:
        mesh_file_for_render = mesh_baker(args.save_folder)
        
    # stage 6, render gif
    # todo fix: if init folder unclear, it maybe mistake rendering
    if args.do_render:
        if mesh_file_for_render and os.path.exists(mesh_file_for_render):
            mesh_file_for_render = mesh_file_for_render
        else:
            baked_fld_list = sorted(glob(args.save_folder + '/view_*/bake/mesh.obj'))
            mesh_file_for_render = baked_fld_list[-1] if len(baked_fld_list)>=1 else args.save_folder+'/mesh.obj'
            assert os.path.exists(mesh_file_for_render), f"{mesh_file_for_render} file not found"
            
        print("Rendering 3d file:", mesh_file_for_render)
        
        if 'gif_renderer' in locals():
            gif_renderer(
                mesh_file_for_render,
                gif_dst_path = os.path.join(args.save_folder, 'output.gif'),
            )
