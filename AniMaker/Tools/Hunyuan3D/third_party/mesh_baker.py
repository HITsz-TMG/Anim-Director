import os, sys, time, traceback
print("sys path insert", os.path.join(os.path.dirname(__file__), "dust3r"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dust3r"))

import cv2
import numpy as np
from PIL import Image, ImageSequence
from einops import rearrange
import torch

from infer.utils import seed_everything, timing_decorator
from infer.utils import get_parameter_number, set_parameter_grad_false

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo

from third_party.gen_baking import back_projection
from third_party.dust3r_utils import infer_warp_mesh_img
from svrm.ldm.vis_util import render_func


class MeshBaker:
    def __init__(
        self, 
        align_model = "third_party/weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        device = "cuda:0", 
        align_times = 1,
        iou_thresh = 0.8,
        save_memory = False
    ):
        self.device = device
        self.save_memory = save_memory
        self.align_model = AsymmetricCroCo3DStereo.from_pretrained(align_model)
        self.align_model = self.align_model if save_memory else self.align_model.to(device)
        self.align_times = align_times
        self.align_model.eval()
        self.iou_thresh = iou_thresh
        set_parameter_grad_false(self.align_model)
        print('baking align model', get_parameter_number(self.align_model))
    
    def align_and_check(self, src, dst, align_times=3):
        try:
            st = time.time()
            best_baking_flag = False
            best_aligned_image = aligned_image = src
            best_info = {'match_num': 1000, "mask_iou": self.iou_thresh-0.1}
            for i in range(align_times):
                aligned_image, info = infer_warp_mesh_img(aligned_image, dst, self.align_model, vis=False)
                aligned_image = Image.fromarray(aligned_image)
                print(f"{i}-th time align process, mask-iou is {info['mask_iou']}")
                if info['mask_iou'] > best_info['mask_iou']:
                    best_aligned_image, best_info = aligned_image, info
                if info['mask_iou'] < self.iou_thresh:
                    break
            # print(f"Best Baking Info:{best_info['mask_iou']}")
            best_baking_flag = best_info['mask_iou'] > self.iou_thresh
            return best_aligned_image, best_info, best_baking_flag
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
            return None, None, None
        
    @timing_decorator("baking mesh")
    def __call__(self, *args, **kwargs):
        if self.save_memory:
            self.align_model = self.align_model.to(self.device)
            torch.cuda.empty_cache()
            res = self.call(*args, **kwargs)
            self.align_model = self.align_model.to("cpu")
        else:
            res = self.call(*args, **kwargs)
        torch.cuda.empty_cache()
        return res
    
    def call(self, save_folder, force=False, front='auto', others=['180°'], align_times=3, seed=0):
        obj_path         = os.path.join(save_folder, "mesh.obj")
        raw_texture_path = os.path.join(save_folder, "texture.png")
        views_pil        = os.path.join(save_folder, "views.jpg")
        views_gif        = os.path.join(save_folder, "views.gif")
        cond_pil         = os.path.join(save_folder, "img_nobg.png")

        if os.path.exists(views_pil):
            views_pil = Image.open(views_pil)
            views = rearrange(np.asarray(views_pil, dtype=np.uint8), '(n h) (m w) c -> (n m) h w c', n=3, m=2)
            views = [Image.fromarray(views[idx]).convert('RGB') for idx in [0,2,4,5,3,1]] 
            cond_pil = Image.open(cond_pil).resize((512,512))
        elif os.path.exists(views_gif):
            views_gif_pil = Image.open(views_gif)
            views = [img.convert('RGB') for img in ImageSequence.Iterator(views_gif_pil)]
            cond_pil, views = views[0], views[1:]
        else:
            raise FileNotFoundError("views file not found")
        
        others = [int(x.replace("°", "")) for x in others]
        
        if len(others)==0:
            rendered_views = render_func(obj_path, elev=0, n_views=1)
        elif len(others)==1 and others[0]==180:
            rendered_views = render_func(obj_path, elev=0, n_views=2)
        else:
            rendered_views = render_func(obj_path, elev=0, n_views=6)
            
        print(f"Need baking views are {others}")
        others = [0] + others
        
        seed_everything(seed)
        
        for ele_idx, ele in enumerate([0, 60, 120, 180, 240, 300]):
            
            if ele not in others: continue
            
            print(f"\n Baking view ele_{ele} ...")
            
            if ele == 0:
                if front == 'input image' or front == 'auto':
                    aligned_cond, cond_info, _ = self.align_and_check(cond_pil, rendered_views[0], align_times=self.align_times)
                    if cond_info is None: continue
                    aligned_cond.convert("RGB").save(save_folder + f'/aligned_cond.jpg')
                    if front == 'input image':
                        aligned_img, info = aligned_cond, cond_info
                        print("Using input image to bake front view")
        
                if front == 'multi-view front view' or front == 'auto':
                    aligned_img, info, _ = self.align_and_check(views[0], rendered_views[0], align_times=self.align_times)
                    if info is None: continue
                    aligned_img.save(save_folder + f'/aligned_{ele}.jpg')
                    print("Using multi-view front view image to bake front view")
                
                if front == 'auto' and info['mask_iou'] < cond_info['mask_iou']:
                    print("Auto using Cond Image to bake front view")
                    aligned_img, info = aligned_cond, cond_info
                    
                need_baking = info['mask_iou'] > self.iou_thresh
                
            else:
                aligned_img, info, need_baking = self.align_and_check(
                    views[ele//60], 
                    rendered_views[min(ele//60, len(others)-1)], 
                    align_times=self.align_times
                )
                if info is None: continue
                aligned_img.save(save_folder + f'/aligned_{ele}.jpg')

            try:
                if need_baking or force:
                    st = time.time()
                    view1_res = back_projection(
                        obj_file = obj_path,
                        init_texture_file = raw_texture_path,
                        front_view_file = aligned_img,
                        dst_dir = os.path.join(save_folder, f"view_{ele_idx}"),
                        render_resolution = aligned_img.size[0], 
                        uv_resolution = 1024,
                        views = [[0, ele]],
                        device = self.device
                    )
                    print(f"view_{ele_idx} elevation_{ele} baking finished at {time.time() - st}")
                    obj_path = os.path.join(save_folder, f"view_{ele_idx}/bake/mesh.obj")
                    raw_texture_path = os.path.join(save_folder, f"view_{ele_idx}/bake/texture.png")
                else:
                    print(f"Skip view_{ele_idx} elevation_{ele} baking")
            except Exception as err:
                print(err)
                continue

        print("\nBaking Finished\n")
        return obj_path
    

if __name__ == "__main__":
    baker = MeshBaker()
    obj_path = baker("./outputs/test")
    print(obj_path)