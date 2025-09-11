import os
import gc
import torch
from Tools.Hunyuan3D.Hunyuan3D import Hunyuan3D

class Script2Character:
    def __init__(self, script_data, story_dir):
        """
        Initializes the Script2Character generator.

        Args:
            script_data (dict): The loaded script data containing character information.
            story_dir (str): The base directory for the current story's results.
        """
        self.script_data = script_data
        self.story_dir = story_dir
        self.output_dir = os.path.join(self.story_dir, 'imgs/characters')
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_characters(self):
        """
        Generates images for each character defined in the script.

        Returns:
            list: A list of dictionaries, each containing information about a generated character image.
        """
        print(f"Starting character image generation...")
        generated_characters_data = []
        try:
            Hunyuan3DGenerator = Hunyuan3D()

            for character in self.script_data.get('characters', []):
                character_name = character.get('name', 'UnknownCharacter')
                character_description = character.get('description', 'No description provided.')
                character_id = character.get('id', None)

                if character_id is None:
                    print(f"Warning: Character '{character_name}' is missing an ID. Skipping.")
                    continue

                formatted_name = character_name.replace(' ', '_')
                save_folder = os.path.join(self.output_dir, formatted_name)
                os.makedirs(save_folder, exist_ok=True)

                print(f"Generating image for character: {character_name}")
                try:
                    Hunyuan3DGenerator.generate(text_prompt=character_description, save_folder=save_folder)
                    print(f"Image for {character_name} saved to {save_folder}")

                    char_img_path = os.path.join(save_folder, "img.jpg")
                    generated_characters_data.append({
                        "character_id": character_id,
                        "character_name": character_name,
                        "character_description": character_description,
                        "output_image": char_img_path if os.path.exists(char_img_path) else None
                    })
                except Exception as e:
                    print(f"Error generating image for character {character_name}: {e}")
                    generated_characters_data.append({
                        "character_id": character_id,
                        "character_name": character_name,
                        "character_description": character_description,
                        "output_image": None,
                        "error": str(e)
                    })

        except Exception as e:
            print(f"Failed to initialize or use Hunyuan3DGenerator: {e}")
        finally:
            # Cleanup GPU memory
            if 'Hunyuan3DGenerator' in locals():
                del Hunyuan3DGenerator
            torch.cuda.empty_cache()
            gc.collect()
            print("Character generation finished, resources released.")

        return generated_characters_data

