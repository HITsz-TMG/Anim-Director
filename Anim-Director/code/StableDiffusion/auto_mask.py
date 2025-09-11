import json
import base64
import requests
import os

class AutoMask:
    def __init__(self, seg_url: str):
        self.seg_url = seg_url

    @staticmethod
    def image_to_base64(image_path: str) -> str:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def save_encoded_images(b64_images: list, base_path: str, prefix: str):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for i, b64_image in enumerate(b64_images):
            output_path = os.path.join(base_path, f'{prefix}{i}.png')
            with open(output_path, 'wb') as image_file:
                image_file.write(base64.b64decode(b64_image))

    def submit_post(self, data: dict):
        return requests.post(self.seg_url, data=json.dumps(data))

    def process_image(self, image_path: str, dino_text_prompt: str):
        if not os.path.exists(image_path):
            print(f"Error: The file {image_path} does not exist.")
            return
        
        directory, base_name = os.path.split(image_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        base_output_path = os.path.join(directory, base_name_no_ext)
        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)
        
        data = {
            "sam_model_name": "sam_vit_b_01ec64.pth",
            "input_image": self.image_to_base64(image_path),
            "sam_positive_points": [],
            "sam_negative_points": [],
            "dino_enabled": True,
            #"dino_model_name": "GroundingDINO_SwinT_OGC (694MB)",
            "dino_model_name": "GroundingDINO_SwinB (938MB)",
            "dino_text_prompt": dino_text_prompt,
        }

        response = self.submit_post(data)
        response_data = response.json()

        categories = {
            'blended_images': f"{dino_text_prompt.replace(' ', '_')}_blended_",
            'masks': f"{dino_text_prompt.replace(' ', '_')}_mask_",
            'masked_images': f"{dino_text_prompt.replace(' ', '_')}_masked_"
        }

        for category, prefix in categories.items():
            images = response_data.get(category, [])
            self.save_encoded_images(images, base_output_path, prefix)
        
        print(f"AutoMask Mission for {image_path} Done.")

# Usage
# if __name__ == '__main__':
#     AutoMasker = AutoMask(seg_url=r'http://127.0.0.1:7860/sam/sam-predict')
#     AutoMasker.process_image(image_path='', dino_text_prompt='')
