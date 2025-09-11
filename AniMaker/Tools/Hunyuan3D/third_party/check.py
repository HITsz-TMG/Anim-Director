import os
import sys
import io

def check_bake_available():
    is_ok = os.path.exists("Tools/Hunyuan3D/third_party/weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt/model.safetensors")
    is_ok = is_ok and os.path.exists("Tools/Hunyuan3D/dust3r")
    is_ok = is_ok and os.path.exists("Tools/Hunyuan3D/dust3r/dust3r")
    is_ok = is_ok and os.path.exists("Tools/Hunyuan3D/dust3r/croco/models")
    if is_ok:
        print("Baking is avaliable")
        print("Baking is avaliable")
        print("Baking is avaliable")
    else:
        print("Baking is unavailable, please download related files in README")
        print("Baking is unavailable, please download related files in README")
        print("Baking is unavailable, please download related files in README")
    return is_ok



if __name__ == "__main__":
    
    check_bake_available()
    