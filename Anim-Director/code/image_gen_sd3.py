import os
import re
import time
import json
import base64
from PIL import Image
from io import BytesIO
from tool.gpt import GPT
from tool.imgur import Imgur
from tool.image_processor import ImageProcessor
from tool.sd3_api import StableDiffusionAPI

class ImageGenerator:
    def __init__(self, story_list, result_file, gpt_organization, gpt_api_key, imgur_client_id, imgur_client_secret, imgur_access_token, imgur_refresh_token, imgur_album_id, proxy):
        self.story_list = story_list
        self.result_file = result_file
        self.gpt_organization = gpt_organization
        self.gpt_api_key = gpt_api_key
        self.imgur_client_id = imgur_client_id
        self.imgur_client_secret = imgur_client_secret
        self.imgur_access_token = imgur_access_token
        self.imgur_refresh_token = imgur_refresh_token
        self.imgur_album_id = imgur_album_id
        self.proxy = proxy
        self.results = {}


    def setup_proxy(self, proxy):
        try:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
            http_proxy_set = os.environ.get('HTTP_PROXY') == proxy
            https_proxy_set = os.environ.get('HTTPS_PROXY') == proxy
            if http_proxy_set and https_proxy_set:
                print("Proxy setup successful.")
            else:
                print("Proxy setup failed. Please check the proxy settings.")
        except Exception as e:
            print(f"An error occurred while setting up the proxy: {e}")


    def save_results(self):
        with open(self.result_file, 'w') as file:
            json.dump(self.results, file, indent=4)


    def Scene2Image(self, id):
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        
        scene2imagine_data = {}
        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        sd3_api = StableDiffusionAPI()
        uploader = Imgur(self.imgur_client_id, self.imgur_client_secret, self.imgur_access_token, self.imgur_refresh_token)
        processor = ImageProcessor()
        save_dir = f'code/result/image/sd3/{id}/Scenes'

        story2scene_answer = self.results[str(id)]['story2scene']['final_answer']
        scene_numbers = [int(num) for num in re.findall(r"Scene (\d+)", story2scene_answer)]
        scene_num = max(scene_numbers) if scene_numbers else 0
        segment_num = scene_num * 2
        script_prompt = self.results[str(id)]['segment2prompt']['final_answer']
        parts = re.split(r'\s*(Characters:|Settings:|Scenes:)\s*', script_prompt, flags=re.IGNORECASE)
        character_part = ""
        for i, part in enumerate(parts):
            if "scenes:" in part.lower():
                if i + 1 < len(parts):
                    character_part = parts[i + 1]
                    break
        character_features = dict(re.findall(r"(\w+[\w\s']*):\s*(.+?)\s*(?=\n\w+[\w\s']*|$)", character_part))
        # print(f"character_features:\n{character_features}")
        scene_part = ""
        for i, part in enumerate(parts):
            if "scenes:" in part.lower():
                if i + 1 < len(parts):
                    scene_part = parts[i + 1]
                    break

        print(f"******************No.{id} Scene2Image Begin******************")

        pattern = re.compile(r'(Scene\s+\d+\s+Segment\s+\d+)')
        parts_with_delimiter = pattern.split(scene_part)
        scene_lines = []
        for i in range(1, len(parts_with_delimiter), 2):
            scene_lines.append(parts_with_delimiter[i] + parts_with_delimiter[i + 1])
        
        scene2imagine_data = self.results.get(str(id), {}).get('scene2image', {})
        scene2imagine_data['segment_num'] = segment_num
        for i, line in enumerate(scene_lines[0:segment_num], start=1):
            print(f"******************No.{id} Scene2Image Segment{i} Begin******************")
            if line.strip(): 
                sceneno_desc = line.split(":")
                sceneno = sceneno_desc[0].strip().strip("-").strip()
                description = sceneno_desc[1].strip().strip("-").strip()
                parts = description.strip().split(']')
                names = parts[0].strip('[').split(',')
                names = [name.strip() for name in names]
                # place = parts[1].strip('[').strip() 
                scene = re.sub(r'[\s/\\n]+$', '', parts[2].strip())

                for character, feature in character_features.items():
                    pattern = fr"({character})\s*\(([^)]+)\)"
                    scene = re.sub(pattern, fr"\1 ({feature})", scene, count=1)
                prompt = "stock illustration style, minimalist style, " + scene
                
                for j in range(2): # Generate twice and take the best
                    print(f"******************No.{id} Scene2Image Segment{i} Round{j+1} Begin******************")
                    black_margin_flag = 1
                    while(black_margin_flag):
                        try:
                            # scene2image step1: Initial generation + image selection
                            available_image_url = []
                            for k in range(4):
                                os.makedirs(save_dir, exist_ok=True)
                                save_path = os.path.join(save_dir, sceneno.replace(' ', '_') + f'_Round_{j+1}_{k+1}' + '.jpg')
                                sd3_api.generate_image(save_path, prompt)
                                available_image_url.append(uploader.upload_image(save_dir), self.imgur_album_id)['link']

                            scene2imagine_step1_question = f"The discription for the illustration is: '" + scene + f"' The available illustrations for choice are labeled from image 1 to image 4 in order, among which each should conform strictly to the description. The key question is: Which image most accurately reflects the characters and the description for the illustration? Note that: 1. Top priority: Check the number of characters image by image from image 1 to image 4. The number of characters in the illustration must be {len(names)}, and each character({str(names)}) must only be depicted once. It is not permissible for two identical characters to appear in the illustration. 2. Illustrations where the sizes of human characters and animals characters are excessively disproportionate, deviating significantly from reality (such as dogs/birds being as tall as humans), should not be choosed. 3. Among the images that meet the first two requirements, select the image that best matches the description for the illustration.\nPlease give me a definite answer('The answer is image x') and then the analysis."
                            scene2imagine_step1_image_url = available_image_url
                            valid_task_id = False
                            while not valid_task_id:
                                scene2imagine_step1_answer = gpt_api.query(prompt=scene2imagine_step1_question, image_urls=scene2imagine_step1_image_url) # Chatgpt Choose 
                                print(f"scene2imagine_step1_answer:\n{scene2imagine_step1_answer}")
                                image_id_match = re.search("The answer is image (\d+)", scene2imagine_step1_answer, re.IGNORECASE)
                                if image_id_match:
                                    image_id = int(image_id_match.group(1))
                                    if 0 <= (image_id - 1) < len(available_image_url):
                                        image_url = available_image_url[image_id - 1]
                                        valid_task_id = True
                                    else:
                                        print("Invalid image ID, trying again...")
                                else:
                                    print("No image ID found, trying again...")

                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, sceneno.replace(' ', '_') + f'_Round_{j+1}' + '.jpg')
                            processor.download_image(image_url, save_path)
                            # If there are black edges, regenerate
                            if(processor.has_black_borders(save_path)):
                                print("Black Margin Detected, Regenerating...")
                                continue
                            else:
                                black_margin_flag = 0
                        except Exception as e:
                            print(f"Error: {e}. Regenerating...")
                            continue
                    if j == 0:
                        scene2imagine_data[sceneno] = {} 
                    scene2imagine_data[sceneno][str(j+1)] = {'available_url': available_image_url, "choice": scene2imagine_step1_answer, "url": image_url}
                    
                # scene2image step2: Pick one of two ultimately
                scene2imagine_step2_question = "The detailed description for the illustration is provided as follows: '" + scene + f"'. You are to choose between Image 1 and Image 2, which are the available options for this illustration. Each image must adhere strictly to the description provided. The primary question to address is: Which image most accurately reflects both the characters and the description of the illustration?\nPlease follow these conditions step by step in your analysis: 1.Check the number of characters in image 1 and in image 2. The number of characters in the illustration must be {len(names)}, and each character must only be depicted once. It is not permissible for two identical characters to appear in the illustration, and no character is allowed to appear in the background of the image. In other words, the selected image must include all the human and animal characters from Image 1, each appearing only once. 2.Avoid selecting any images with black or white margins. 3.Do not choose illustrations where the sizes of human and animal characters are excessively disproportionate, deviating significantly from reality (for example, dogs or birds as tall as humans). 4.Select the image that best aligns with the description of the illustration.\nAfter considering these conditions, please provide a definitive answer ('The answer is Image 1' or 'The answer is Image 2') and then your analysis."
                scene2imagine_step2_image_url = [scene2imagine_data[sceneno]['1']['url'], scene2imagine_data[sceneno]['2']['url']]
                valid_task_id = False
                while not valid_task_id:
                    scene2imagine_step2_answer = gpt_api.query(prompt=scene2imagine_step2_question, image_urls=scene2imagine_step2_image_url) # Chatgpt Choose 
                    print(f"scene2imagine_step2_answer:\n{scene2imagine_step2_answer}")
                    image_id_match = re.search("The answer is image (\d+)", scene2imagine_step2_answer, re.IGNORECASE)
                    if image_id_match:
                        image_id = int(image_id_match.group(1))
                        if image_id == 1:
                            final_url = scene2imagine_data[sceneno]['1']['url']                        
                            valid_task_id = True
                        elif image_id == 2:
                            final_url = scene2imagine_data[sceneno]['2']['url']                        
                            valid_task_id = True 
                        else:
                            print("Invalid image ID, trying again...")
                    else:
                        print("No image ID found, trying again...")
                
                save_path = os.path.join(save_dir, sceneno.replace(' ', '_') + '.jpg')
                while True:
                    try:
                        processor.download_image(final_url, save_path)
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}. Restarting the iteration...")
                        time.sleep(300)
                        continue
                scene2imagine_data[sceneno] = {'scene': scene, 'prompt': prompt, "final_choice": scene2imagine_step2_answer, "final_url": final_url, '1': scene2imagine_data[sceneno]['1'], '2': scene2imagine_data[sceneno]['2']}
                self.results[str(id)]['scene2image'] = scene2imagine_data  
                with open(self.result_file, 'r') as file:
                    self.results = json.load(file)
                    
        self.results[str(id)]['scene2image'] = scene2imagine_data  
        print(f"******************No.{id} Scene2Image Completed******************")


    def main(self):
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        for idNo in story_list:
            self.Scene2Image(idNo)
            self.save_results()


if __name__ == '__main__':
    story_list = [32, 34, 54] # modify as you want
    result_file = 'code/result/script.json'
    gpt_organization = "your gpt_organization"
    gpt_api_key = "your gpt_api_key"
    imgur_client_id = 'your imgur_client_id'
    imgur_client_secret = 'your imgur_client_secret'
    imgur_access_token = 'your imgur_access_token'
    imgur_refresh_token = 'your imgur_refresh_token'
    imgur_album_id = 'your imgur_album_id'
    proxy = 'your proxy'

    generator = ImageGenerator(story_list, result_file, gpt_organization, gpt_api_key, imgur_client_id, imgur_client_secret, imgur_access_token, imgur_refresh_token, imgur_album_id, proxy)
    generator.main()