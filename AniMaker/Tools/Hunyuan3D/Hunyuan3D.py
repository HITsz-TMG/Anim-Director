import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
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

class Hunyuan3D:
    def __init__(self, 
                 use_lite=False,
                 save_memory=False,
                 mv23d_cfg_path="Tools/Hunyuan3D/svrm/configs/svrm.yaml",
                 mv23d_ckt_path="Tools/Hunyuan3D/weights/svrm/svrm.safetensors",
                 text2image_path="Tools/Hunyuan3D/weights/hunyuanDiT"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lite = use_lite
        self.save_memory = save_memory
        
        # Fixed parameters
        self.max_faces_num = 90000
        self.do_texture_mapping = True
        self.do_render = True
        self.t2i_seed = 42
        self.t2i_steps = 25
        self.gen_seed = 0
        self.gen_steps = 50
        #self.do_bake = False
        self.do_bake = True
        self.bake_align_times = 3
        
        # Initialize models
        st = time.time()
        self.rembg_model = Removebg()
        self.image_to_views_model = Image2Views(
            device=self.device, 
            use_lite=self.use_lite,
            save_memory=self.save_memory
        )
        
        self.views_to_mesh_model = Views2Mesh(
            mv23d_cfg_path, 
            mv23d_ckt_path, 
            self.device, 
            use_lite=self.use_lite,
            save_memory=self.save_memory
        )
        
        self.text_to_image_model = Text2Image(
            pretrain=text2image_path,
            device=self.device, 
            save_memory=self.save_memory
        )
        
        if self.do_bake and BAKE_AVAILEBLE:
            self.mesh_baker = MeshBaker(
                device=self.device,
                align_times=self.bake_align_times
            )
                
        if check_bake_available():
            self.gif_renderer = GifRenderer(device=self.device)
            
        print(f"Init Models cost {time.time()-st}s")
    
    def generate(self, text_prompt, save_folder,):
        """Generate 3D model from text prompt"""
        os.makedirs(save_folder, exist_ok=True)

        # stage 1, text to image
        res_rgb_pil = self.text_to_image_model(
            text_prompt, 
            seed=self.t2i_seed,  
            steps=self.t2i_steps
        )
        res_rgb_pil.save(os.path.join(save_folder, "img.jpg"))

        # stage 2, remove back ground
        res_rgba_pil = self.rembg_model(res_rgb_pil)
        res_rgba_pil.save(os.path.join(save_folder, "img_nobg.png"))

        # stage 3, image to views
        (views_grid_pil, cond_img), view_pil_list = self.image_to_views_model(
            res_rgba_pil,
            seed=self.gen_seed,
            steps=self.gen_steps
        )
        views_grid_pil.save(os.path.join(save_folder, "views.jpg"))

        # # stage 4, views to mesh
        # self.views_to_mesh_model(
        #     views_grid_pil, 
        #     cond_img, 
        #     seed=self.gen_seed,
        #     target_face_count=self.max_faces_num,
        #     save_folder=save_folder,
        #     do_texture_mapping=self.do_texture_mapping
        # )
        
        # # stage 5, baking
        # mesh_file_for_render = None
        # if self.do_bake and BAKE_AVAILEBLE:
        #     mesh_file_for_render = self.mesh_baker(save_folder)
            
        # # stage 6, render gif
        # if self.do_render:
        #     if mesh_file_for_render and os.path.exists(mesh_file_for_render):
        #         mesh_file_for_render = mesh_file_for_render
        #     else:
        #         baked_fld_list = sorted(glob(save_folder + '/view_*/bake/mesh.obj'))
        #         mesh_file_for_render = baked_fld_list[-1] if len(baked_fld_list)>=1 else save_folder+'/mesh.obj'
        #         assert os.path.exists(mesh_file_for_render), f"{mesh_file_for_render} file not found"
                
        #     print("Rendering 3d file:", mesh_file_for_render)
            
        #     if hasattr(self, 'gif_renderer'):
        #         self.gif_renderer(
        #             mesh_file_for_render,
        #             gif_dst_path=os.path.join(save_folder, 'output.gif'),
        #         )

# # Example usage
# if __name__ == "__main__":
#     text_prompt = "A young boy with a slender build; Golden hair, neatly parted; Large, innocent eyes; A bright blue tunic; Brown trousers; Small black boots."
#     save_folder = "./outputs/tmp/"
#     Hunyuan3DGenerator = Hunyuan3D()
#     Hunyuan3DGenerator.generate(text_prompt=text_prompt, save_folder=save_folder)
