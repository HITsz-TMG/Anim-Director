import os
import re
import gc
import torch
from PIL import Image
from Tools.Flux.Flux import FluxGenerator
from Tools.gemini_api import GeminiAPI
from Tools.deepseek_api import DeepSeekAPI

class Script2Environment:
    def __init__(self, script_data, story_dir, deepseek_api_key, gemini_api_keys, gemini_proxy):
        """
        Initializes the Script2Environment generator.

        Args:
            script_data (dict): The loaded script data containing environment information.
            story_dir (str): The base directory for the current story's results.
            deepseek_api_key (str): API key for DeepSeek.
            gemini_api_keys (list): List of API keys for Gemini.
            gemini_proxy (str): Proxy setting for Gemini API.
        """
        self.script_data = script_data
        self.story_dir = story_dir
        self.output_dir = os.path.join(self.story_dir, 'imgs/environments')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize APIs and Generator
        self.deepseek_api = DeepSeekAPI(api_key=deepseek_api_key)
        self.gemini_api = GeminiAPI(api_keys=gemini_api_keys, proxy=gemini_proxy)
        self.flux_generator = FluxGenerator()
        self.num_images_per_environment = 3 # Configurable number of images per env

    def _generate_simplified_prompt(self, description, api_name):
        """Generates a simplified prompt using the specified API."""
        prompt_template = f"""
        I need a background image for an animation.
        ### {description} ###
        The description between ### ### is too complex with too many elements.
        Please simplify it to 15 words, retaining only key elements.
        Directly give the simplified prompt without any guide words or markdown.
        """
        if api_name == "DeepSeek":
            return self.deepseek_api.generate_from_text(prompt_template)
        elif api_name == "Gemini":
            return self.gemini_api.generate_from_text(prompt_template)
        else:
            raise ValueError(f"Unknown API name: {api_name}")

    def _select_best_image(self, description, image_paths):
        """Uses Gemini to select the best image based on the description."""
        if not image_paths:
            return None, None # No images to select from

        image_selection_prompt = f"""
        I have generated {len(image_paths)} images for an environment with this description:
        "{description}"

        Please analyze these image paths and tell me which one best matches the environment description.
        Just respond with the number of the best image (1, 2, or 3) without explanation.
        """
        try:
            print("Asking Gemini to select the best environment image...")
            best_image_response = self.gemini_api.generate_multimodal(prompt=image_selection_prompt, image_paths=image_paths)
            match = re.search(r'\b[1-3]\b', best_image_response)
            if match:
                best_image_idx = int(match.group()) - 1
                if 0 <= best_image_idx < len(image_paths):
                    print(f"Gemini selected image {best_image_idx + 1}")
                    return best_image_idx, image_paths[best_image_idx]
                else:
                    print(f"Invalid image index from Gemini: {best_image_idx + 1}. Response: {best_image_response}")
            else:
                print(f"Could not extract a valid image number from Gemini's response: {best_image_response}")
        except Exception as e:
            print(f"Error during Gemini image selection: {str(e)}")

        # Fallback to the first image if selection fails
        print("Falling back to selecting the first image.")
        return 0, image_paths[0]


    def generate_environments(self):
        """
        Generates images for each environment defined in the script.

        Returns:
            list: A list of dictionaries, each containing information about a generated environment.
        """
        print(f"Starting environment image generation...")
        all_environments_data = []

        try:
            for environment_idx, environment in enumerate(self.script_data.get('environments', [])):
                environment_name = environment.get('name', f'UnknownEnvironment_{environment_idx}')
                environment_description = environment.get('description', 'No description provided.')
                environment_id = environment.get('id', None)

                if environment_id is None:
                    print(f"Warning: Environment '{environment_name}' is missing an ID. Skipping.")
                    continue

                formatted_name = environment_name.replace(' ', '_')
                save_folder = os.path.join(self.output_dir, formatted_name)
                os.makedirs(save_folder, exist_ok=True)

                print(f"\nProcessing environment: {environment_name} (ID: {environment_id})")
                env_data = {
                    "environment_id": environment_id,
                    "environment_name": environment_name,
                    "environment_description": environment_description,
                    "generations": [],
                    "selected_image": None
                }

                generated_image_paths = []
                for img_idx in range(self.num_images_per_environment):
                    try:
                        # Alternate APIs for prompt generation
                        api_name = "DeepSeek" if img_idx < 2 else "Gemini"
                        simplified_prompt = self._generate_simplified_prompt(environment_description, api_name)

                        formatted_prompt = f"Background image, Medium shot: {simplified_prompt} Minimalism style, 2D animation."
                        print(f"  Generating image {img_idx+1}/{self.num_images_per_environment} using {api_name} prompt: {formatted_prompt}")

                        # Generate image with Flux
                        image = self.flux_generator.generate(formatted_prompt)

                        # Save the output image
                        output_filename = f"{formatted_name}_{api_name}_{img_idx+1}.png"
                        output_path = os.path.join(save_folder, output_filename)
                        self.flux_generator.save_image(image, output_path)
                        generated_image_paths.append(output_path)
                        print(f"  Saved image to {output_path}")

                        # Track generation information
                        env_data["generations"].append({
                            "prompt": formatted_prompt,
                            "api_used": api_name,
                            "output_image": output_path
                        })

                    except Exception as e:
                        print(f"  Error generating image {img_idx+1} for environment {environment_name}: {str(e)}")

                # Select the best image
                best_image_idx, best_image_path = self._select_best_image(environment_description, generated_image_paths)

                if best_image_path:
                    main_image_filename = f"{formatted_name}.png"
                    main_image_path = os.path.join(save_folder, main_image_filename)
                    try:
                        selected_image = Image.open(best_image_path)
                        selected_image.save(main_image_path)
                        print(f"  Selected image {best_image_idx + 1} saved as main reference: {main_image_path}")
                        env_data["selected_image"] = {
                            "index": best_image_idx + 1,
                            "path": main_image_path,
                            "original_path": best_image_path
                        }
                    except Exception as e:
                         print(f"  Error saving selected image {best_image_path} to {main_image_path}: {e}")
                else:
                    print(f"  No images were generated successfully for environment {environment_name}, cannot select best.")

                all_environments_data.append(env_data)

        except Exception as e:
            print(f"An error occurred during the environment generation process: {e}")
        finally:
            # Cleanup GPU memory for Flux
            if hasattr(self, 'flux_generator'):
                del self.flux_generator
            torch.cuda.empty_cache()
            gc.collect()
            print("\nEnvironment generation finished, resources released.")

        return all_environments_data

