import os
import requests
import base64  # Add base64 import
from io import BytesIO
from Tools.gemini_api import GeminiAPI
from Tools.deepseek_api import DeepSeekAPI
from openai import OpenAI

class Script2Scene:
    def __init__(self, script_data, story_dir, deepseek_api_key, gemini_api_keys, gemini_proxy, openai_api_key):
        self.script_data = script_data
        self.story_dir = story_dir
        self.deepseek_api = DeepSeekAPI(api_key=deepseek_api_key)
        self.gemini_api = GeminiAPI(api_keys=gemini_api_keys, proxy=gemini_proxy)
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.output_dir = os.path.join(self.story_dir, 'imgs/scenes')
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_prompts_per_clip = 5  # Generate 5 prompts for each clip

    def generate_image_with_gpt(self, prompt, input_images, output_path):
        """
        Generate an image using OpenAI's gpt-image-1 model with multiple input images
        """
        try:
            if not self.openai_client:
                raise ValueError("OpenAI API key not provided")
                
            # Open all input images
            images = [open(img_path, "rb") for img_path in input_images]
            
            result = self.openai_client.images.edit(
                model="gpt-image-1",
                image=images,
                prompt=prompt,
                size="1024x1534",
                quality="standard",
            )
            
            # Get base64 encoded image and decode it
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # Save the image to a file
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            print(f"Image saved to {output_path}")
            
            # Close all opened files
            for img in images:
                img.close()
                
            return output_path
                
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None
            
    def generate_scene_prompts(self):
        """
        Generates reference image prompts for scenes where characters or environment change.
        """
        scene_generation_results = []
        prev_chars = None
        prev_env = None

        # Process all clips across all scenes
        for scene_idx, scene in enumerate(self.script_data['script']):
            scene_name = f"scene_{scene_idx+1}"

            for clip_idx, clip in enumerate(scene['clips']):
                # Get current clip's characters and environment
                current_chars = set(clip['characters'])
                current_env = scene['environment']
                clip_id = clip['id']

                # Check if we need to generate a reference image
                # Generate ref image for first clip or when characters/environment change
                if prev_chars is None or prev_env is None or current_chars != prev_chars or current_env != prev_env:
                    print(f"Change detected in clip {clip_id} - generating reference image prompts")

                    # Create output folder for this scene (just for organization)
                    clip_folder = os.path.join(self.output_dir, f"{scene_name}/clip_{clip_id}")
                    os.makedirs(clip_folder, exist_ok=True)

                    # Initialize clip data for tracking
                    clip_data = {
                        "scene_name": scene_name,
                        "clip_id": clip_id,
                        "scene_description": clip['description'],
                        "generations": []
                    }

                    # Collect character images
                    character_images = []
                    character_names = []
                    for char_id in clip['characters']:
                        # Find character by ID in the characters list
                        character = next((c for c in self.script_data['characters'] if str(c['id']) == char_id), None)
                        if not character:
                            print(f"Warning: Character with ID {char_id} not found in character list")
                            continue

                        character_name = character['name']
                        formatted_name = character_name.replace(' ', '_')
                        char_img_path = os.path.join(self.story_dir, f'imgs/characters/{formatted_name}/img.jpg')
                        if os.path.exists(char_img_path):
                            character_images.append(char_img_path)
                            character_names.append(character_name)
                        else:
                            print(f"Warning: Character image not found at {char_img_path}")

                    # Get environment image by ID
                    env_id = scene['environment']
                    environment = next((e for e in self.script_data['environments'] if e['id'] == env_id), None)
                    if not environment:
                        print(f"Warning: Environment with ID {env_id} not found")
                        continue

                    env_name = environment['name']
                    formatted_env_name = env_name.replace(' ', '_')
                    env_img_path = os.path.join(self.story_dir, f'imgs/environments/{formatted_env_name}/{formatted_env_name}.png')

                    if not os.path.exists(env_img_path):
                        print(f"Warning: Environment image not found at {env_img_path}")
                        continue

                    # Create input images list
                    input_images = character_images + [env_img_path]
                    print(f"Input images for clip {clip_id}: {input_images}")

                    prompt_template = f"""
                    Generate a prompt for image generation that describes this clip:
                    "{clip['description']}"

                    The prompt must mention these characters: {', '.join(character_names)} and this environment: {env_name}.
                    Note that the prompt should depict a static image rather than a dynamic action sequence. And it should be as simple as possible. Do not include dialogue.

                    Format the prompt like:
                    "16:9, image generation: [Character 1] (in image 1) (and [Character 2] in (in image 2)) is/are doing something in the [Environment] (in image x)"
                    e.g.
                    1."16:9, image generation: The little boy (in image 1) walks towards a rose under the frost-kissed glass dome in the icy desert night (in image 2), looking worriedly."
                    2."16:9, image generation: The little boy (in image 1) met the fox (in image 2) in the rose garden (in image 3), and they exchanged warm greetings."
                    """

                    # Generate multiple clip prompts using different APIs
                    for prompt_idx in range(self.num_prompts_per_clip):
                        try:
                            if prompt_idx < 4:
                                api_name = "DeepSeek"
                                formatted_prompt = self.deepseek_api.generate_from_text(prompt_template)
                            else:
                                api_name = "Gemini"
                                formatted_prompt = self.gemini_api.generate_from_text(prompt_template)

                            print(f"Generated prompt for clip {clip_id}, prompt {prompt_idx+1} using {api_name}:\n{formatted_prompt}")
                            
                            # Generate image using gpt-image-1 if OpenAI API key is provided
                            output_image = "Waiting to be Generated"
                            if self.openai_client:
                                image_filename = f"clip_{clip_id}_prompt_{prompt_idx+1}.png"
                                image_path = os.path.join(clip_folder, image_filename)
                                output_image = self.generate_image_with_gpt(formatted_prompt, input_images, image_path) or output_image
                            
                            # Track generation information
                            generation_info = {
                                "prompt": formatted_prompt,
                                "api_used": api_name,
                                "input_images": input_images,
                                "output_image": output_image,
                            }
                            clip_data["generations"].append(generation_info)

                        except Exception as e:
                            print(f"Error generating prompt {prompt_idx+1} for clip {clip_id}: {str(e)}")

                    scene_generation_results.append(clip_data)
                else:
                    print(f"No changes in clip {clip_id} - skipping reference image prompt generation")

                # Update previous clip info for next comparison
                prev_chars = current_chars
                prev_env = current_env

        return scene_generation_results
