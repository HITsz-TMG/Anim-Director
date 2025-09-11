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
from tool.midjourney_api import MidJourneyAPI
from StableDiffusion.auto_mask import AutoMask

class ImageGenerator:
    def __init__(self, story_list, result_file, gpt_organization, gpt_api_key, mj_api_key, imgur_client_id, imgur_client_secret, imgur_access_token, imgur_refresh_token, imgur_album_id, proxy):
        self.story_list = story_list
        self.result_file = result_file
        self.gpt_organization = gpt_organization
        self.gpt_api_key = gpt_api_key
        self.mj_api_key = mj_api_key
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


    def Character2Image(self, id):
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        
        character2imagine_data = {}
        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        mj_api = MidJourneyAPI(self.mj_api_key)

        script_prompt = self.results[str(id)]['segment2prompt']['final_answer']
        parts = re.split(r'\s*(Characters:|Settings:|Scenes:)\s*', script_prompt, flags=re.IGNORECASE)
        character_part = ""
        for i, part in enumerate(parts):
            if "characters:" in part.lower():
                if i + 1 < len(parts):
                    character_part = parts[i + 1]
                    break

        print(f"******************No.{id} Character2Image Begin******************") # MJ x Chatgpt
        character_lines = character_part.split("\n")[0:]  # Intercept the Character part
        for line in character_lines:
            while True:
                try:
                    if line.strip():  # Check if the line is not empty
                        name_desc = line.split(":")
                        name = name_desc[0].strip().strip("-").strip()
                        description = name + ": " + name_desc[1].strip().strip("-").strip().replace("anthropomorphic ", "")
                        
                        # Generate two kinds: front view and back view
                        success = False
                        while not success:
                            prompt = "Concept character art, front view and back view, " + description + " full length body view, white background. --niji 6 --s 50"
                            available_task_id = []
                            available_image_url = []
                            try:
                                original_task_id, original_image_url = mj_api.draw(prompt) # MJ Imagine 1
                            except Exception as e:
                                print(f"Imaging failed due to: {e}. Retrying...")
                                time.sleep(10)
                                continue
                            for i in range(1, 5):
                                task_id, image_url = mj_api.enlarge(original_task_id, str(i)) # MJ Upscale
                                available_task_id.append(task_id)
                                available_image_url.append(image_url)
                            character2imagine_question = "The description of the image is: '" + description + f"' The images are labeled from image 1 to image 4 in order, all conforming strictly to the description. The question is: Which picture most accurately reflects the established character in the textual description?\nPlease give me a definite answer('The answer is image x') and then the analysis. Note: The first priority is that the chosen image must contain exactly two full-body portraits of the same character, one showing the front view and the other showing the back view."
                            valid_task_id = False
                            while not valid_task_id:
                                character2imagine_answer = gpt_api.query(prompt=character2imagine_question, image_urls=available_image_url) # Chatgpt Choose     
                                print(f"character2imagine_answer:\n{character2imagine_answer}")
                                image_id_match = re.search("The answer is image (\d+)", character2imagine_answer, re.IGNORECASE)
                                if image_id_match:
                                    image_id = int(image_id_match.group(1))
                                    if 0 <= (image_id - 1) < len(available_task_id):
                                        task_id = available_task_id[image_id - 1]
                                        image_url = available_image_url[image_id - 1]
                                        valid_task_id = True
                                    else:
                                        print("Invalid image ID, trying again...")
                                else:
                                    print("No image ID found, trying again...")
                            save_dir = os.path.join(os.path.join('code/result/image/mj', str(id)), 'Characters')
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, name.replace(' ', '_') + '.jpg') # Download Image
                            download_processor = ImageProcessor() 
                            download_processor.download_image(image_url, save_path)
                            processor = ImageProcessor(save_path)
                            try:
                                front_image_path, back_image_path = processor.split_image()  # Split Image
                                success = True
                            except Exception as e:
                                print(f"Splitting failed due to: {e}. Retrying...")
                        uploader = Imgur(self.imgur_client_id, self.imgur_client_secret, self.imgur_access_token, self.imgur_refresh_token)
                        original_front_image_url = uploader.upload_image(front_image_path, self.imgur_album_id)["link"]
                        original_back_image_url = uploader.upload_image(back_image_path, self.imgur_album_id)["link"]
                        front_image_path = processor.resize_image(front_image_path)
                        back_image_path = processor.resize_image(back_image_path)
                        print(f"Split images saved at {front_image_path} and {back_image_path}")
                        front_image_url = uploader.upload_image(front_image_path, self.imgur_album_id)["link"]
                        back_image_url = uploader.upload_image(back_image_path, self.imgur_album_id)["link"]
                        character2imagine_data[name] = {'prompt': description, 'available_id': available_task_id, 'available_url': available_image_url, "choice": character2imagine_answer, "id": task_id, "url": image_url, "original_front_url": original_front_image_url, "original_back_url": original_back_image_url, "front_url": front_image_url, "back_url": back_image_url}
                    break
                except Exception as e:
                    print(f"An error occurred: {e}. Restarting the iteration...")
                    continue
        self.results[str(id)]['character2image'] = character2imagine_data   
        print(f"******************No.{id} Character2Image Completed******************")


    def Setting2Image(self, id):
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        
        setting2imagine_data = {}
        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        mj_api = MidJourneyAPI(self.mj_api_key)

        script_prompt = self.results[str(id)]['segment2prompt']['final_answer']
        parts = re.split(r'\s*(Characters:|Settings:|Scenes:)\s*', script_prompt, flags=re.IGNORECASE)
        setting_part = ""
        for i, part in enumerate(parts):
            if "settings:" in part.lower():
                if i + 1 < len(parts):
                    setting_part = parts[i + 1]
                    break

        print(f"******************No.{id} Setting2Image Begin******************")
        setting_lines = setting_part.split("\n")[0:]  # Intercept the Setting Section
        for line in setting_lines:
            while True:
                try:
                    if line.strip():
                        place_desc = line.split(":")
                        place = place_desc[0].strip().strip("-").strip()
                        description = place + ": " + place_desc[1].strip().strip("-").strip()
                        prompt = "stock illustration style, close up shot, front view, 30° look up composition, natural light, " + description + " --niji 6 --s 50 --ar 3:2"
                        available_task_id = []
                        available_image_url = []
                        original_task_id, original_image_url = mj_api.draw(prompt) # MJ Imagine 1
                        for i in range(1, 5):
                            task_id, image_url = mj_api.enlarge(original_task_id, str(i)) # MJ Upscale
                            available_task_id.append(task_id)
                            available_image_url.append(image_url)
                        setting2imagine_question = "The description of the image is: '" + description + f"' The images are labeled from image 1 to image 4 in order, all conforming strictly to the description. The key question is: Which picture most accurately reflects the established setting in the textual description?\nPlease give me a definite answer('The answer is image x') and then the analysis."
                        valid_task_id = False
                        while not valid_task_id:
                            setting2imagine_answer = gpt_api.query(prompt=setting2imagine_question, image_urls=available_image_url) # Chatgpt Choose 
                            print(f"setting2imagine_answer:\n{setting2imagine_answer}")
                            image_id_match = re.search("The answer is image (\d+)", setting2imagine_answer, re.IGNORECASE)
                            if image_id_match:
                                image_id = int(image_id_match.group(1))
                                if 0 <= (image_id - 1) < len(available_task_id):
                                    task_id = available_task_id[image_id - 1]
                                    image_url = available_image_url[image_id - 1]
                                    valid_task_id = True
                                else:
                                    print("Invalid image ID, trying again...")
                            else:
                                print("No image ID found, trying again...")
                        save_dir = os.path.join(os.path.join('code/result/image/mj', str(id)), 'Settings')
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, place.replace(' ', '_') + '.jpg')
                        download_processor = ImageProcessor() 
                        download_processor.download_image(image_url, save_path)
                        uploader = Imgur(self.imgur_client_id, self.imgur_client_secret, self.imgur_access_token, self.imgur_refresh_token)
                        image_url = uploader.upload_image(save_path, self.imgur_album_id)["link"]
                        setting2imagine_data[place] = {'prompt': description, 'available_id': available_task_id, 'available_url': available_image_url, "choice": setting2imagine_answer, "id": task_id, "url": image_url}
                    break
                except Exception as e:
                    print(f"An error occurred: {e}. Restarting the iteration...")
                    continue
        self.results[str(id)]['setting2image'] = setting2imagine_data   
        print(f"******************No.{id} Setting2Image Completed******************")


    def Scene2Image(self, id):
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        
        scene2imagine_data = {}
        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        mj_api = MidJourneyAPI(self.mj_api_key)
        uploader = Imgur(self.imgur_client_id, self.imgur_client_secret, self.imgur_access_token, self.imgur_refresh_token)
        processor = ImageProcessor()
        save_dir = f'code/result/image/mj/{id}/Scenes'

        story2scene_answer = self.results[str(id)]['story2scene']['final_answer']
        scene_numbers = [int(num) for num in re.findall(r"Scene (\d+)", story2scene_answer)]
        scene_num = max(scene_numbers) if scene_numbers else 0
        segment_num = scene_num * 2
        script_prompt = self.results[str(id)]['segment2prompt']['final_answer']
        parts = re.split(r'\s*(Characters:|Settings:|Scenes:)\s*', script_prompt, flags=re.IGNORECASE)
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
        for i, line in enumerate(scene_lines[0:segment_num], start=1):
            print(f"******************No.{id} Scene2Image Segment{i} Begin******************")
            if line.strip(): 
                sceneno_desc = line.split(":")
                sceneno = sceneno_desc[0].strip().strip("-").strip()
                description = sceneno_desc[1].strip().strip("-").strip()
                parts = description.strip().split(']')
                names = parts[0].strip('[').split(',')
                names = [name.strip() for name in names]
                place = parts[1].strip('[').strip() 
                scene = re.sub(r'[\s/\\n]+$', '', parts[2].strip())  

                # scene2image step1: Select the character cutout (front and back)
                names_str = ', '.join(names)
                scene2imagine_step1_iamge = [self.results[str(id)]['setting2image'][place]['url']]
                scene2imagine_step1_question = "I want to deaw a 2D illustration for the scene: " + scene + " The other parts of the illustration have already been created except for the characters, and is shown in the given image. I want to complete the characters for this illustration, and I need your help about the orientation of the characters in the illustration. The characters of the illustration are: " + names_str + ". Please decide the orientation relative to the audience (between front and back) for each character in the scene, considering only their interactions with the environment and with each other. In other words, If the orientation of a character is front, the illustration should display the front of the character; And if it is back, the illustration should display the back of the character. For instance, if a door in the illustration is facing the audience, and Tom is facing the door, then Tom's orientation would be back. And if Kitty is talking to Tom at the moment, her orientation would also be back, to maintain consistency with Tom in the 2D illustration. If not necessary for the story, the front orientation is preferred to highlight the character's demeanor and actions. Please give me a direct answer and then the analysis. Your answer should strictly follow the format: 'Character1: orientation, Character2: orientation, etc. Analysis:...' And the name of character before the ':' should be among the character list: '" + names_str + "'. For example: If the character list is 'Tom, Tom's friend Kitty', then your answer can be: 'Tom: front, Tom's friend Kitty: front. Analysis:...'"
                scene2imagine_step1_answer = gpt_api.query(prompt=scene2imagine_step1_question, image_urls=scene2imagine_step1_iamge) 
                print(f"scene2imagine_step1_answer:\n{scene2imagine_step1_answer}")
                try:
                    orientations_str, _ = scene2imagine_step1_answer.split('Analysis')[0].strip(), scene2imagine_step1_answer.split('Analysis')[1].strip()
                    orientation_info = {re.sub(r'^[^\w]*(.*?)[^\w]*$', r'\1', item.split(':')[0].strip()): re.sub(r'^[^\w]*(.*?)[^\w]*$', r'\1', item.split(':')[1].strip()) for item in orientations_str.split(',')}
                except Exception as e:
                    orientation_info = {name: 'front' for name in names}
                
                character_urls = []
                character_paths = []
                name_to_orientation_suffix = {}
                for name in names:
                    try:
                        orientation = orientation_info.get(name, 'front').strip()
                        orientation_suffix = 'back' if 'back' in orientation else 'front'
                        name_filename = name.replace(' ', '_')
                        character_url = self.results[str(id)]['character2image'][name][orientation_suffix + '_url']
                        character_path = f"code/result/image/mj/{id}/Characters/{name_filename}_{orientation_suffix}.jpg"
                    except Exception as e:
                        name_filename = name.replace(' ', '_')
                        character_url = self.results[str(id)]['character2image'][name]['front_url']
                        character_path = f"code/result/image/mj/{id}/Characters/{name_filename}_front.jpg"
                    character_urls.append(character_url)
                    character_paths.append(character_path)  
                    name_to_orientation_suffix[name] = orientation_suffix          

                if (len(character_urls) > 1): # Use Segment Anything to change the background of the character cutout to transparent color to prevent white edges
                    sceneno_filename = sceneno.replace(' ', '_')
                    character_path = processor.stitch_images(character_paths, f"code/result/image/mj/{id}/Scenes/Characters/{sceneno_filename}.jpg")
                    AutoMasker = AutoMask(seg_url=r'http://127.0.0.1:7860/sam/sam-predict')
                    os.environ['HTTP_PROXY'] = '' # AutoMask needn't a proxy
                    os.environ['HTTPS_PROXY'] = ''
                    AutoMasker.process_image(image_path=character_path, dino_text_prompt='All Characters') # Generate Masks
                    self.setup_proxy(proxy)
                    base_path = os.path.join(os.path.dirname(character_path), sceneno.replace(' ', '_'))
                    while True:
                        try:
                            masked_url_0 = uploader.upload_image(os.path.join(base_path, 'All_Characters_masked_0.png'), imgur_album_id)['link'] # Imgur Masks Upload
                            masked_url_1 = uploader.upload_image(os.path.join(base_path, 'All_Characters_masked_1.png'), imgur_album_id)['link']
                            masked_url_2 = uploader.upload_image(os.path.join(base_path, 'All_Characters_masked_2.png'), imgur_album_id)['link']
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the iteration...")
                            time.sleep(300)
                            continue
                    available_masked_url = [masked_url_0, masked_url_1, masked_url_2]
                    scene2imagine_step1plus_question = "Among the three images, which image features the most complete portrayal of the characters, whose edges are not missing due to cropping?\nPlease give me a definite answer('The answer is image x') and then the analysis."
                    valid_task_id = False
                    while not valid_task_id:
                        scene2imagine_step1plus_answer = gpt_api.query(prompt=scene2imagine_step1plus_question, image_urls=available_masked_url) # Chatgpt Choose Mask
                        print(f"scene2imagine_step1plus_answer:\n{scene2imagine_step1plus_answer}")
                        masked_id_match = re.search("The answer is image (\d+)", scene2imagine_step1plus_answer, re.IGNORECASE)
                        if masked_id_match:
                            masked_id = int(masked_id_match.group(1))
                            if 1 <= masked_id <= 3:
                                valid_task_id = True
                            else:
                                print("Invalid mask ID, trying again...")
                        else:
                            print("No mask ID found, trying again...")
                    character_path = os.path.join(base_path, f'All_Characters_masked_{masked_id-1}.png')
                    while True:
                        try:
                            character_url = uploader.upload_image(character_path, imgur_album_id) # Imgur Characters Upload
                            character_url = character_url['link']
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the iteration...")
                            time.sleep(300)
                            continue
                else:
                    character_url = character_urls[0]
                    character_path = character_paths[0]
                setting_url = self.results[str(id)]['setting2image'][place]['url']
                place_filename = place.replace(' ', '_')
                setting_path = (f"code/result/image/mj/{id}/Settings/{place_filename}.jpg")

                character = ''
                for name in names:
                    orientation = orientation_info.get(name, 'front').strip()
                    orientation_suffix = '(back oriented)' if 'back' in orientation else '(front oriented)'
                    original_character_prompt = self.results[str(id)]['character2image'][name]['prompt']
                    modified_character_prompt = original_character_prompt.replace(name, f"{name}{orientation_suffix}")
                    character += ' ' + modified_character_prompt
                setting = ' ' + self.results[str(id)]['setting2image'][place]['prompt']
                if(len(names) == 1):
                    prompt = setting_url + " " + "stock illustration style, minimalist style," + character + setting + " " + scene + " --cref " + character_url + " --v 6.0 --s 50 --ar 16:9 --iw 1.5 --no black or white margins" # v6
                elif(len(names) == 2):
                    layout = names[0] + ' is on the left of ' + names[1]
                    prompt = character_url + " " + setting_url + " " + "stock illustration style, minimalist style," + character + setting + " " + layout + '. '+ scene + " --v 6.0 --s 50 --ar 16:9 --iw 1.5 --no black or white margins" # v6
                else:
                    prompt = character_url + " " + setting_url + " " + "stock illustration style, minimalist style," + character + setting + " " + scene + " --v 6.0 --s 50 --ar 16:9 --iw 1.5 --no black or white margins" # v6
                
                for j in range(2): # Generate twice and take the best
                    print(f"******************No.{id} Scene2Image Segment{i} Round{j+1} Begin******************")
                    black_margin_flag = 1
                    while(black_margin_flag):
                        try:
                            # scene2image step2: Initial generation + image selection
                            available_task_id = []
                            available_image_url = []
                            original_task_id, original_image_url = mj_api.draw(prompt) # MJ Imagine 1
                            for k in range(1, 5):
                                task_id, image_url = mj_api.enlarge(original_task_id, str(k)) # MJ Upscale
                                available_task_id.append(task_id)
                                available_image_url.append(image_url)

                            scene2imagine_step2_question = f"The characters for the illustration are shown in the first image: {str(names)}. The background setting for the illustration is shown in the second image. The discription for the illustration is: '" + scene + f"' The available illustrations for choice are labeled from image 3 to image 6 in order, among which each should conform strictly to the description. The key question is: Which image most accurately reflects the characters、background setting and the description for the illustration? Note that: 1. Top priority: Check the number of characters image by image from image 3 to image 6. The number of characters in the illustration must be {len(names)}, and each character({str(names)}) must only be depicted once. It is not permissible for two identical characters to appear in the illustration. 2. Illustrations where the sizes of human characters and animals characters are excessively disproportionate, deviating significantly from reality (such as dogs/birds being as tall as humans), should not be choosed. 3. Among the images that meet the first two requirements, select the image that best matches the description for the illustration.\nPlease give me a definite answer('The answer is image x') and then the analysis."
                            scene2imagine_step2_image_url = [character_url, setting_url] + available_image_url
                            valid_task_id = False
                            while not valid_task_id:
                                scene2imagine_step2_answer = gpt_api.query(prompt=scene2imagine_step2_question, image_urls=scene2imagine_step2_image_url) # Chatgpt Choose 
                                print(f"scene2imagine_step2_answer:\n{scene2imagine_step2_answer}")
                                image_id_match = re.search("The answer is image (\d+)", scene2imagine_step2_answer, re.IGNORECASE)
                                if image_id_match:
                                    image_id = int(image_id_match.group(1))
                                    if 0 <= (image_id - 3) < len(available_task_id):
                                        task_id = available_task_id[image_id - 3]
                                        image_url = available_image_url[image_id - 3]
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
                        
                    # scene2image step3 + scene2image step4: Inpaint the Characters in the Scene one by one
                    mask_task_id = task_id # Always point to the image id that needs to be partially modified
                    mask_image_url = image_url
                    if(len(names) > 1):
                        AutoMasker = AutoMask(seg_url=r'http://127.0.0.1:7860/sam/sam-predict')
                        for name in names:
                            while True:
                                try: 
                                    # scene2image step3: Masks generation + selection
                                    original_prompt = self.results[str(id)]['character2image'][name]['prompt'] # Extract the core description of the character (the first sentence after the colon)
                                    dino_text_question = "The character description is as follows: \"" + original_prompt + "\" Please distill the above character description information: Retain only the identity, the age and the upper garment for human character; Retain only the animal kind and the color for animal character. For example, for the character description: \"Mr. Green: Friendly, middle-aged neighbor. Stocky build, greying hair, kind green eyes, in a green sweater, khaki pants, and garden gloves.\", you should return \"a middle-aged neighbor wearing a green sweater\" For the character description: \"A loyal and joyful dog, with a shiny golden fur coat that glistens under the sun and deep brown eyes filled with curiosity and kindness.\", you should return: \"a shiny golden dog\". If any infomation requested above is not given, just omit it. Please give the abbreviated character description directly without including any lead-in words or punctuation."
                                    dino_text_prompt = gpt_api.query(prompt=dino_text_question)
                                    
                                    os.environ['HTTP_PROXY'] = '' # AutoMask needn't a proxy
                                    os.environ['HTTPS_PROXY'] = ''                   
                                    AutoMasker.process_image(image_path=save_path, dino_text_prompt=dino_text_prompt) # Masks generation
                                    self.setup_proxy(proxy)
                                    # print(dino_text_prompt)

                                    base_path = os.path.join(save_dir, sceneno.replace(' ', '_')) + f'_Round_{j+1}'
                                    while True:
                                        try:
                                            masked_url_0 = uploader.upload_image(os.path.join(base_path, dino_text_prompt.replace(' ', '_') + '_masked_0.png'), imgur_album_id)['link'] # Imgur Masks Upload
                                            masked_url_1 = uploader.upload_image(os.path.join(base_path, dino_text_prompt.replace(' ', '_') + '_masked_1.png'), imgur_album_id)['link']
                                            masked_url_2 = uploader.upload_image(os.path.join(base_path, dino_text_prompt.replace(' ', '_') + '_masked_2.png'), imgur_album_id)['link']
                                            break
                                        except Exception as e:
                                            print(f"An error occurred: {e}. Restarting the iteration...")
                                            time.sleep(300)
                                            continue
                                    available_masked_url = [masked_url_0, masked_url_1, masked_url_2]

                                    scene2imagine_step3_question = "These three images all correspond to the same character description: " + dino_text_prompt +". Please select the image where the character is portrayed most completely, with no edges missing due to cropping.\nGive me a definite answer('The answer is image x.') and then the analysis. If none of the three images accurately portrays the correct character alone (either depicting multiple characters or a totally different character), explain the mistake. For instance, a cat should not be depicted as a dog, and a middle-aged individual should not be depicted as a child."
                                    valid_task_id = False
                                    retry_times = 0
                                    while not valid_task_id:
                                        scene2imagine_step3_answer = gpt_api.query(prompt=scene2imagine_step3_question, image_urls=available_masked_url) # Chatgpt Choose Mask
                                        print(f"scene2imagine_step3_answer:\n{scene2imagine_step3_answer}")
                                        masked_id_match = re.search("The answer is image (\d+)", scene2imagine_step3_answer, re.IGNORECASE)
                                        if masked_id_match:
                                            masked_id = int(masked_id_match.group(1))
                                            if 1 <= masked_id <= 3:
                                                valid_task_id = True
                                            else:
                                                print("Invalid mask ID, trying again...")
                                        else:
                                            print("No mask ID found, trying again...")
                                            retry_times += 1
                                            if retry_times > 10:
                                                break
                                            time.sleep(10)
                                    if retry_times > 10:
                                        continue
                                    mask_path = os.path.join(base_path, dino_text_prompt.replace(' ', '_') + f'_mask_{masked_id-1}.png')
                                    
                                    # scene2image step4: Vary (Region)
                                    mask_prompt = self.results[str(id)]['character2image'][name]['original_' + name_to_orientation_suffix[name] + '_url'] + " " + "stock illustration style, minimalist style " + results[str(id)]['character2image'][name]['prompt'] + " --niji 6 --s 50 --ar 16:9 --iw 1.8" 
                                    img = Image.open(mask_path)
                                    buffer = BytesIO()
                                    img.save(buffer, format="WEBP")
                                    webp_img = buffer.getvalue()
                                    mask_base64_webp = base64.b64encode(webp_img).decode()
                                    region_task_id, region_image_url = mj_api.region(mask_task_id, mask_prompt, mask_base64_webp) # MJ Vary (Region)
                                    
                                    region_available_task_id = []
                                    region_available_image_url = []
                                    for k in range(1, 5):
                                        task_id, image_url = mj_api.enlarge(region_task_id, str(k)) # MJ Upscale
                                        region_available_task_id.append(task_id)
                                        region_available_image_url.append(image_url)
                                    scene2imagine_step4_question = "The characters for the illustration are shown in the first image. The background setting for the illustration is shown in the second image. The discription for the illustration is: '" + scene + f"' The available illustrations for choice are labeled from image 3 to image 6 in order, among which each should conform strictly to the description. The key question is: Which image most accurately reflects the characters、background setting and the description for the illustration? Note that any image with black or white margins must not be choosed.\nPlease give me a definite answer('The answer is image x') and then the analysis. After that, for the selected image, verify that the character's appearance matches the first image, with particular attention to the color of the clothing and the face. If significant inconsistencies are detected (e.g., apparently different clothing colors), generate a notification like 'Character Consistency Failed'"
                                    scene2imagine_step4_image_url = [character_url, setting_url] + region_available_image_url
                                    valid_task_id = False
                                    while not valid_task_id:
                                        scene2imagine_step4_answer = gpt_api.query(prompt=scene2imagine_step4_question, image_urls=scene2imagine_step4_image_url) # Chatgpt Choose     
                                        print(f"scene2imagine_step4_answer:\n{scene2imagine_step4_answer}")
                                        image_id_match = re.search("The answer is image (\d+)", scene2imagine_step4_answer, re.IGNORECASE)
                                        if image_id_match:
                                            image_id = int(image_id_match.group(1))
                                            if 'Character Consistency Failed' in scene2imagine_step4_answer:
                                                print("Character Consistency Failed, trying again...")
                                            elif 0 <= (image_id - 3) < len(available_task_id):
                                                mask_task_id = region_available_task_id[image_id - 3]
                                                mask_image_url = region_available_image_url[image_id - 3]
                                                valid_task_id = True
                                            else:
                                                print("Invalid image ID, trying again...")
                                        else:
                                            print("No image ID found, trying again...")
                                    break
                                except Exception as e:
                                    print(f"An error occurred: {e}. Restarting the iteration...")
                                    continue
                    
                    while True:
                        try:
                            processor.download_image(mask_image_url, save_path)
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the iteration...")
                            time.sleep(300)
                            continue
                    if j == 0:
                        scene2imagine_data[sceneno] = {} 
                    scene2imagine_data[sceneno][str(j+1)] = {'available_id': available_task_id, 'available_url': available_image_url, "pre_choice": scene2imagine_step2_answer, "pre_id": task_id, "pre_url": image_url, "regioned_id": mask_task_id, "regioned_url": mask_image_url}
                    
                # scene2image step5: Pick one of two ultimately
                scene2imagine_step5_question = "The characters featured in the illustration are: " + names_str + ". The initial portrayal of these characters is shown in Image 1. The detailed description for the illustration is provided as follows: '" + scene + f"'. You are to choose between Image 2 and Image 3, which are the available options for this illustration. Each image must adhere strictly to the description provided. The primary question to address is: Which image most accurately reflects both the characters and the description of the illustration?\nPlease follow these conditions step by step in your analysis: 1.Check the number of characters in image 2 and in image 3. The number of characters in the illustration must be {len(names)}, and each character must only be depicted once. It is not permissible for two identical characters to appear in the illustration, and no character is allowed to appear in the background of the image. In other words, the selected image must include all the human and animal characters from Image 1, each appearing only once. 2.Avoid selecting any images with black or white margins. 3.Do not choose illustrations where the sizes of human and animal characters are excessively disproportionate, deviating significantly from reality (for example, dogs or birds as tall as humans). 4.Select the image that best aligns with the description of the illustration.\nAfter considering these conditions, please provide a definitive answer ('The answer is Image 2' or 'The answer is Image 3') and then your analysis."
                scene2imagine_step5_image_url = [character_url, scene2imagine_data[sceneno]['1']['regioned_url'], scene2imagine_data[sceneno]['2']['regioned_url']]
                valid_task_id = False
                while not valid_task_id:
                    scene2imagine_step5_answer = gpt_api.query(prompt=scene2imagine_step5_question, image_urls=scene2imagine_step5_image_url) # Chatgpt Choose 
                    print(f"scene2imagine_step5_answer:\n{scene2imagine_step5_answer}")
                    image_id_match = re.search("The answer is image (\d+)", scene2imagine_step5_answer, re.IGNORECASE)
                    if image_id_match:
                        image_id = int(image_id_match.group(1))
                        if image_id == 2:
                            regioned_id = scene2imagine_data[sceneno]['1']['regioned_id']
                            regioned_url = scene2imagine_data[sceneno]['1']['regioned_url']                        
                            valid_task_id = True
                        elif image_id == 3:
                            regioned_id = scene2imagine_data[sceneno]['2']['regioned_id']
                            regioned_url = scene2imagine_data[sceneno]['2']['regioned_url']                        
                            valid_task_id = True 
                        else:
                            print("Invalid image ID, trying again...")
                    else:
                        print("No image ID found, trying again...")
                
                save_path = os.path.join(save_dir, sceneno.replace(' ', '_') + '.jpg')
                while True:
                    try:
                        processor.download_image(regioned_url, save_path)
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}. Restarting the iteration...")
                        time.sleep(300)
                        continue
                scene2imagine_data[sceneno] = {'prompt': scene, 'character': character_path, 'setting': setting_path, "final_choice": scene2imagine_step5_answer, "regioned_id": regioned_id, "regioned_url": regioned_url, '1': scene2imagine_data[sceneno]['1'], '2': scene2imagine_data[sceneno]['2']}
                self.results[str(id)]['scene2image'] = scene2imagine_data  
                with open(self.result_file, 'r') as file:
                    self.results = json.load(file)
                    
        self.results[str(id)]['scene2image'] = scene2imagine_data  
        print(f"******************No.{id} Scene2Image Completed******************")


    def Transition2Image(self, id):
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        
        transition2imagine_data = {}
        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        mj_api = MidJourneyAPI(self.mj_api_key)
        uploader = Imgur(self.imgur_client_id, self.imgur_client_secret, self.imgur_access_token, self.imgur_refresh_token)
        processor = ImageProcessor()
        save_dir = os.path.join(os.path.join('code/result/image/mj', str(id)), 'Scenes')

        story2scene_answer = self.results[str(id)]['story2scene']['final_answer']
        scene_numbers = [int(num) for num in re.findall(r"Scene (\d+)", story2scene_answer)]
        scene_num = max(scene_numbers) if scene_numbers else 0
        segment_num = scene_num * 2
        script_prompt = self.results[str(id)]['segment2prompt']['final_answer']
        parts = re.split(r'\s*(Characters:|Settings:|Scenes:)\s*', script_prompt, flags=re.IGNORECASE)
        scene_part = ""
        for i, part in enumerate(parts):
            if "scenes:" in part.lower():
                if i + 1 < len(parts):
                    scene_part = parts[i + 1]
                    break

        print(f"******************No.{id} Transition2Image Begin******************")

        pattern = re.compile(r'(Scene\s+\d+\s+Segment\s+\d+)')
        parts_with_delimiter = pattern.split(scene_part)
        scene_lines = []
        for i in range(1, len(parts_with_delimiter), 2):
            scene_lines.append(parts_with_delimiter[i] + parts_with_delimiter[i + 1])
        
        original_names = []
        original_place = ""
        original_scene = ""
        for line in scene_lines[0:segment_num]:
            if line.strip():
                sceneno_desc = line.split(":")
                sceneno = sceneno_desc[0].strip().strip("-").strip()
                parts = sceneno_desc[0].split() 
                scene_part = " ".join(parts[:2])  # Only keep the first two parts, i.e. "Scene X"
                sceneno = scene_part + " transition"
                description = sceneno_desc[1].strip().strip("-").strip()
                parts = description.strip().split(']')
                names = parts[0].strip('[').split(',')
                names = [name.strip() for name in names]
                place = parts[1].strip('[').strip() 
                scene = re.sub(r'[\s/\\n]+$', '', parts[2].strip())

                # transition2image step1: Select the character cutout (front and back) and Propose transition prompt
                if(original_place != "" and place != original_place and not set(original_names).isdisjoint(set(names))):
                    # print(sceneno)
                    transition2imagine_step1_question = "I require the creation of a transitional scene that bridges Scene 1 and Scene 2. Scene 1: " + original_scene + " In Scene 1, we have characters " + str(original_names) + " within the setting of " + original_place + "; Scene 2: " + scene + " In Scene 2, we have characters " + str(names) + " within the setting of " + place + ". Note, the transition scene should depict the relevant characters leaving the location of Scene 1. The characters featured in the transition scene should be a selection from those introduced in Scene 1 and Scene 2, with the stipulation that their names must remain strictly consistent with those mentioned in [] in earlier scenes, tailored to the specific narrative requirements of the plot. The specific verb used depends on the context. Use actions with significant movement whenever possible. The format of your answer should adhere to the following guidelines based on the nature of the transition: for departures from an indoor setting, use 'out of' (e.g., 'CharacterA, CharacterB walk/run/... out of the library'), for entries from an outdoor setting to an indoor one, employ 'into' (e.g., 'CharacterA, CharacterB walk/run/... into the house'), and for movements from one outdoor location to another, use 'from...to...' (e.g., 'CharacterA and CharacterB walk/run/... from the park to the street'). Your answer should be a sentense strictly following the rules above. For example, Amy and Amy's father run out of the kitchen (indoor). For the transitional scene description you are to answer, characters included should be: " + str(set(names).intersection(set(original_names)))
                    transition2imagine_step1_answer = gpt_api.query(prompt=transition2imagine_step1_question) 
                    print(f"transition2imagine_step1_answer:\n{transition2imagine_step1_answer}")

                    character_urls = []
                    character_paths = []
                    transition_names = []
                    processed_names = set()
                    for name in set(names).intersection(set(original_names)):
                        name_filename = name.replace(' ', '_')
                        character_url = self.results[str(id)]['character2image'][name]['front_url']
                        character_path = f"code/result/image/mj/{id}/Characters/{name_filename}_front.jpg"
                        character_urls.append(character_url)
                        character_paths.append(character_path)
                        transition_names.append(name)
                        processed_names.add(name) 
                    # print(character_paths)
                    # print(transition_names)

                    if (len(character_urls) > 1): # Use Segment Anything to change the background of the character cutout to transparent color to prevent white edges
                        sceneno_filename = sceneno.replace(' ', '_')
                        character_path = processor.stitch_images(character_paths, f"code/result/image/mj/{id}/Scenes/Characters/{sceneno_filename}.jpg")
                        AutoMasker = AutoMask(seg_url=r'http://127.0.0.1:7860/sam/sam-predict')
                        os.environ['HTTP_PROXY'] = '' # AutoMask needn't a proxy
                        os.environ['HTTPS_PROXY'] = ''
                        AutoMasker.process_image(image_path=character_path, dino_text_prompt='All Characters') # 生成Mask
                        self.setup_proxy(proxy)
                        base_path = os.path.join(os.path.dirname(character_path), sceneno.replace(' ', '_'))
                        while True:
                            try:
                                masked_url_0 = uploader.upload_image(os.path.join(base_path, 'All_Characters_masked_0.png'), imgur_album_id)['link'] # Imgur Masks Upload
                                masked_url_1 = uploader.upload_image(os.path.join(base_path, 'All_Characters_masked_1.png'), imgur_album_id)['link']
                                masked_url_2 = uploader.upload_image(os.path.join(base_path, 'All_Characters_masked_2.png'), imgur_album_id)['link']
                                break
                            except Exception as e:
                                print(f"An error occurred: {e}. Restarting the iteration...")
                                time.sleep(300)
                                continue
                        available_masked_url = [masked_url_0, masked_url_1, masked_url_2]
                        scene2imagine_step1plus_question = "Among the three images, which image features the most complete portrayal of the characters, whose edges are not missing due to cropping?\nPlease give me a definite answer('The answer is image x') and then the analysis."
                        valid_task_id = False
                        while not valid_task_id:
                            scene2imagine_step1plus_answer = gpt_api.query(prompt=scene2imagine_step1plus_question, image_urls=available_masked_url) # Chatgpt Choose Mask
                            print(f"scene2imagine_step1plus_answer:\n{scene2imagine_step1plus_answer}")
                            masked_id_match = re.search("The answer is image (\d+)", scene2imagine_step1plus_answer, re.IGNORECASE)
                            if masked_id_match:
                                masked_id = int(masked_id_match.group(1))
                                if 1 <= masked_id <= 3:
                                    valid_task_id = True
                                else:
                                    print("Invalid mask ID, trying again...")
                            else:
                                print("No mask ID found, trying again...")
                        character_path = os.path.join(base_path, f'All_Characters_masked_{masked_id-1}.png')
                        while True:
                            try:
                                character_url = uploader.upload_image(character_path, imgur_album_id) # Imgur Characters Upload
                                break
                            except Exception as e:
                                print(f"An error occurred: {e}. Restarting the iteration...")
                                time.sleep(300)
                                continue
                        character_url = character_url['link']
                    else:
                        character_url = character_urls[0]
                        character_path = character_paths[0]
                    setting_url = self.results[str(id)]['setting2image'][original_place]['url']
                    place_filename = original_place.replace(' ', '_')
                    setting_path = (f"code/result/image/mj/{id}/Settings/{place_filename}.jpg")

                    character = ''
                    for name in transition_names:
                        character += ' ' + self.results[str(id)]['character2image'][name]['prompt']
                    setting = ' ' + self.results[str(id)]['setting2image'][original_place]['prompt']
                    if(len(transition_names) == 1):
                        prompt = setting_url + " " + "stock illustration style, minimalist style," + character + setting + " " + transition2imagine_step1_answer + " --cref " + character_url + " --v 6.0 --s 50 --ar 16:9 --iw 1.5 --no black or white margins" # v6
                    else:
                        prompt = character_url + " " + setting_url + " " + "stock illustration style, minimalist style," + character + setting + " " + transition2imagine_step1_answer + " --v 6.0 --s 50 --ar 16:9 --iw 1.8 --no black or white margins" # v6
                    # print(prompt)

                    # transition2image step2: Initial generation + image selection
                    while True:
                        try:
                            available_task_id = []
                            available_image_url = []
                            original_task_id, original_image_url = mj_api.draw(prompt) # MJ Imagine 1
                            for i in range(1, 5):
                                task_id, image_url = mj_api.enlarge(original_task_id, str(i)) # MJ Upscale
                                available_task_id.append(task_id)
                                available_image_url.append(image_url)
                            transition2imagine_step2_question = f"The characters for the illustration are shown in the first image: {str(transition_names)}. The background setting for the illustration is shown in the second image. The discription for the illustration is: '" + transition2imagine_step1_answer + f"' The available illustrations for choice are labeled from image 3 to image 6 in order, among which each should conform strictly to the description. The key question is: Which image most accurately reflects the characters、background setting and the description for the illustration? Note that any image with black or white margins must not be choosed.\nPlease give me a definite answer('The answer is image x') and then the analysis."
                            transition2imagine_step2_image_url = [character_url, setting_url] + available_image_url
                            valid_task_id = False
                            while not valid_task_id:
                                transition2imagine_step2_answer = gpt_api.query(prompt=transition2imagine_step2_question, image_urls=transition2imagine_step2_image_url) # Chatgpt Choose     
                                print(f"transition2imagine_step2_answer:\n{transition2imagine_step2_answer}")
                                image_id_match = re.search("The answer is image (\d+)", transition2imagine_step2_answer, re.IGNORECASE)
                                if image_id_match:
                                    image_id = int(image_id_match.group(1))
                                    if 0 <= (image_id - 3) < len(available_task_id):
                                        task_id = available_task_id[image_id - 3]
                                        image_url = available_image_url[image_id - 3]
                                        valid_task_id = True
                                    else:
                                        print("Invalid image ID, trying again...")
                                else:
                                    print("No image ID found, trying again...")
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the loop...")
                            continue
                        else:
                            break

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, sceneno.replace(' ', '_') + '.jpg')
                    while True:
                        try:
                            processor.download_image(image_url, save_path)
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the iteration...")
                            time.sleep(300)
                            continue
                    # transition2image step3 + transition2image step4: Inpaint the Characters in the Scene one by one
                    while True:
                        try:
                            processor.download_image(image_url, save_path)
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the iteration...")
                            time.sleep(300)
                            continue
                    mask_task_id = task_id # Always point to the image id that needs to be partially modified
                    mask_image_url = image_url
                    if(len(transition_names) > 1):
                        AutoMasker = AutoMask(seg_url=r'http://127.0.0.1:7860/sam/sam-predict')
                        for name in transition_names: 
                            while True:
                                try:
                                    # transition2image step3: Masks generation + selection                    
                                    original_prompt = self.results[str(id)]['character2image'][name]['prompt'] # Extract the core description of the character (the first sentence after the colon)
                                    dino_text_question = "The character description is as follows: \"" + original_prompt + "\" Please distill the above character description information: Retain only the identity, the age and the upper garment for human character; Retain only the animal kind and the color for animal character. For example, for the character description: \"Mr. Green: Friendly, middle-aged neighbor. Stocky build, greying hair, kind green eyes, in a green sweater, khaki pants, and garden gloves.\", you should return \"a middle-aged neighbor wearing a green sweater\" For the character description: \"A loyal and joyful dog, with a shiny golden fur coat that glistens under the sun and deep brown eyes filled with curiosity and kindness.\", you should return: \"a shiny golden dog\". If any infomation requested above is not given, just omit it. Please give the abbreviated character description directly without including any lead-in words or punctuation."
                                    dino_text_prompt = gpt_api.query(prompt=dino_text_question)
                                    
                                    os.environ['HTTP_PROXY'] = '' # AutoMask needn't a proxy
                                    os.environ['HTTPS_PROXY'] = ''                   
                                    AutoMasker.process_image(image_path=save_path, dino_text_prompt=dino_text_prompt) # Masks generation
                                    self.setup_proxy(proxy)
                                    # print(dino_text_prompt)

                                    base_path = os.path.join(save_dir, sceneno.replace(' ', '_'))
                                    masked_url_0 = uploader.upload_image(os.path.join(base_path, dino_text_prompt.replace(' ', '_') + '_masked_0.png'), imgur_album_id)['link'] # Imgur Masks Upload
                                    masked_url_1 = uploader.upload_image(os.path.join(base_path, dino_text_prompt.replace(' ', '_') + '_masked_1.png'), imgur_album_id)['link']
                                    masked_url_2 = uploader.upload_image(os.path.join(base_path, dino_text_prompt.replace(' ', '_') + '_masked_2.png'), imgur_album_id)['link']
                                    available_masked_url = [masked_url_0, masked_url_1, masked_url_2]

                                    transition2image_step3_question = "These three images all correspond to the same character description: " + dino_text_prompt +". Please select the image where the character is portrayed most completely, with no edges missing due to cropping.\nGive me a definite answer('The answer is image x.') and then the analysis. If none of the three images accurately portrays the correct character alone (either depicting multiple characters or a totally different character), explain the mistake. For instance, a cat should not be depicted as a dog, and a middle-aged individual should not be depicted as a child."
                                    valid_task_id = False
                                    retry_times = 0
                                    while not valid_task_id:
                                        transition2image_step3_answer = gpt_api.query(prompt=transition2image_step3_question, image_urls=available_masked_url) # Chatgpt Choose Mask
                                        print(f"transition2image_step3_answer:\n{transition2image_step3_answer}")
                                        masked_id_match = re.search("The answer is image (\d+)", transition2image_step3_answer, re.IGNORECASE)
                                        if masked_id_match:
                                            masked_id = int(masked_id_match.group(1))
                                            if 1 <= masked_id <= 3:
                                                valid_task_id = True
                                            else:
                                                print("Invalid mask ID, trying again...")
                                        else:
                                            print("No mask ID found, trying again...")
                                            retry_times += 1
                                            if retry_times > 10:
                                                break
                                            time.sleep(10)
                                    if retry_times > 10:
                                        continue
                                    mask_path = os.path.join(base_path, dino_text_prompt.replace(' ', '_') + f'_mask_{masked_id-1}.png')
                                    
                                    # transition2image step4: Vary (Region)
                                    mask_prompt = self.results[str(id)]['character2image'][name]['original_front_url'] + " " + "stock illustration style, minimalist style " + self.results[str(id)]['character2image'][name]['prompt'] + " --niji 6 --s 50 --ar 16:9 --iw 1.8" 
                                    img = Image.open(mask_path)
                                    buffer = BytesIO()
                                    img.save(buffer, format="WEBP")
                                    webp_img = buffer.getvalue()
                                    mask_base64_webp = base64.b64encode(webp_img).decode()
                                    region_task_id, region_image_url = mj_api.region(mask_task_id, mask_prompt, mask_base64_webp) # MJ Vary (Region)
                                    
                                    region_available_task_id = []
                                    region_available_image_url = []
                                    for k in range(1, 5):
                                        task_id, image_url = mj_api.enlarge(region_task_id, str(k)) # MJ Upscale
                                        region_available_task_id.append(task_id)
                                        region_available_image_url.append(image_url)
                                    transition2image_step4_question = "The characters for the illustration are shown in the first image. The background setting for the illustration is shown in the second image. The discription for the illustration is: '" + transition2imagine_step1_answer + f"' The available illustrations for choice are labeled from image 3 to image 6 in order, among which each should conform strictly to the description. The key question is: Which image most accurately reflects the characters、background setting and the description for the illustration? Note that any image with black or white margins must not be choosed.\nPlease give me a definite answer('The answer is image x') and then the analysis. After that, for the selected image, verify that the character's appearance matches the first image, with particular attention to the color of the clothing and the face. If significant inconsistencies are detected (e.g., apparently different clothing colors), generate a notification like 'Character Consistency Failed'"
                                    transition2image_step4_image_url = [character_url, setting_url] + region_available_image_url
                                    valid_task_id = False
                                    while not valid_task_id:
                                        transition2image_step4_answer = gpt_api.query(prompt=transition2image_step4_question, image_urls=transition2image_step4_image_url) # Chatgpt Choose     
                                        print(f"transition2image_step4_answer:\n{transition2image_step4_answer}")
                                        image_id_match = re.search("The answer is image (\d+)", transition2image_step4_answer, re.IGNORECASE)
                                        if image_id_match:
                                            image_id = int(image_id_match.group(1))
                                            if 'Character Consistency Failed' in transition2image_step4_answer:
                                                print("Character Consistency Failed, trying again...")
                                            elif 0 <= (image_id - 3) < len(available_task_id):
                                                mask_task_id = region_available_task_id[image_id - 3]
                                                mask_image_url = region_available_image_url[image_id - 3]
                                                valid_task_id = True
                                            else:
                                                print("Invalid image ID, trying again...")
                                        else:
                                            print("No image ID found, trying again...")
                                    break
                                except Exception as e:
                                    print(f"An error occurred: {e}. Restarting the iteration...")
                                    continue
                    
                    while True:
                        try:
                            processor.download_image(mask_image_url, save_path)
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the iteration...")
                            time.sleep(300)
                            continue
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, sceneno.replace(' ', '_') + '.jpg')
                    while True:
                        try:
                            processor.download_image(image_url, save_path)
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}. Restarting the iteration...")
                            time.sleep(300)
                            continue

                    transition2imagine_data[sceneno] = {'prompt': transition2imagine_step1_answer, 'character': character_path, 'setting':setting_path, 'available_id': available_task_id, 'available_url': available_image_url, "pre_choice": transition2imagine_step2_answer, "pre_id": task_id, "pre_url": image_url, "regioned_id": mask_task_id, "regioned_url": mask_image_url}
                    self.results[str(id)]['transition2image'] = transition2imagine_data  
                    with open(self.result_file, 'r') as file:
                        self.results = json.load(file) 

                original_names = names
                original_place = place
                original_scene = scene
                    
        self.results[str(id)]['transition2image'] = transition2imagine_data  
        print(f"******************No.{id} Transition2Image Completed******************")


    def main(self):
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        for idNo in story_list:
            self.Character2Image(idNo)
            self.save_results()
            self.Setting2Image(idNo)
            self.save_results()
            self.Scene2Image(idNo)
            self.save_results()
            self.Transition2Image(idNo)
            self.save_results()


if __name__ == '__main__':
    story_list = [32, 34, 54] # modify as you want
    result_file = 'code/result/script.json'
    gpt_organization = "your gpt_organization"
    gpt_api_key = "your gpt_api_key"
    mj_api_key = "your mj_api_key"
    imgur_client_id = 'your imgur_client_id'
    imgur_client_secret = 'your imgur_client_secret'
    imgur_access_token = 'your imgur_access_token'
    imgur_refresh_token = 'your imgur_refresh_token'
    imgur_album_id = 'your imgur_album_id'
    proxy = 'your proxy'

    generator = ImageGenerator(story_list, result_file, gpt_organization, gpt_api_key, mj_api_key, imgur_client_id, imgur_client_secret, imgur_access_token, imgur_refresh_token, imgur_album_id, proxy)
    generator.main()