import os
import sys
import json
from Tools.gemini_api import GeminiAPI
# from ..Tools.gemini_r1_api import GeminiR1API
from Tools.deepseek_api import DeepSeekAPI

class Story2Script:
    def __init__(self, story, deepseek_api_key, gemini_api_keys, gemini_proxy):
        self.story = story
        self.gemini_api = GeminiAPI(
            api_keys=gemini_api_keys, 
            proxy=gemini_proxy
        )
        self.deepseek_api = DeepSeekAPI(
            api_key=deepseek_api_key
        )
        self.schema = """{
            "characters": [
                {
                    "id": 1,
                    "name": "[Character's full name] eg. Mia",
                    "description": "[Character summary (age/gender/build)]; [Hair style & color]; [Facial features]; [Upper garment style & color]; [Lower garment style & color]; [Footwear style & color] eg. 12-year-old girl with petite frame; Chestnut hair in messy braid; Round face with freckled cheeks; Oversized mustard yellow sweater; Patched navy corduroy overalls; Scuffed brown ankle boots."
                }
            ],
            "environments": [
                {
                    "id": 1,
                    "name": "[Unique environment name] eg. Moonlit Alley",
                    "description": "[A vivid and cinematic depiction of the scene, encompassing the time of day, weather conditions, primary elements, colors, and ambient details. The description should be sufficiently detailed to translate effortlessly into a visual masterpiece, with distinct visual components for the background, midground, and foreground.] eg. Moonlit cobblestone alley with glistening rain puddles, Victorian gas lamps casting amber pools of light, distant church bell tolling midnight."
                }
            ],
            "script": [
                {
                    "scene": 1,
                    "environment": 1,
                    "clips": [
                        {
                            "id": 1,
                            "characters": ["1"],
                            "description": "A concise and detailed description of the movie clip, including character actions, facial expressions, dialogue, interactions with the environment, and camera shots.] eg. Jimmy sits alone in a dim room, eyes fixed on a worn photograph. A soft sigh escapes his lips as he gently traces the faces with a finger. \\"I miss you,\\" he whispers, his voice tinged with sorrow. The camera slowly zooms in, focusing on his face, capturing the sadness in his eyes."
                        }
                    ]
                }
            ]
        }"""
        self.rules_for_long = """
        Rules:
        1. Include 2 or 3 or 4 #environments# and 1 or 2 or 3 #characters# and detailed #description# related.
        2. Animals can serve as #characters#. They may possess the ability to talk or exhibit anthropomorphic behaviors, but their #character descriptions# should not include any anthropomorphic appearance elements. Instead, the descriptions should be based on their natural appearances, providing sufficient detail to enable accurate illustrations of them. Plants cannot serve as #characters#.
        3. Any #environment# should be the background where part of the storyboard takes place. Do not include any character in the #environment description# part, nor any characteristics connected to the plot. #Environment description# Should be suitable for use as a storyboard background, easy to depict, and not overshadow the main story.
        4. New #scene# only when #environment# changes, and any #environment# can be used in different #scenes#.
        5. Each #scene# contains sequential clips with:
        - #Environment# reference
        - Present #characters#
        - Visual actions (camera moves/transitions)
        - Natural dialogue (when speaking)
        6. At most 2 #characters# can be included in a #clip#.
        7. Consider adding some #clips# of entering/exiting the #environment# at the beginning/end of the #scene#. eg1."Jimmy stood up with a look of disappointment, turned back and walked away, the camera zoomed out." eg2."Jimmy walks from the left side of the picture and greets the fox." Note that the previous/next #environment# should not be included in the #clip description#.
        8. Use present tense action descriptions.
        9. Return the content in json format directly without any guide words or markdown.
        """
        self.check_for_long = """
        Evaluate this animation script against the following criteria:

        1. Schema validation:
        - Verify the script contains #characters#, #environments#, and #script#
        - Each #character# must have id, name, and detailed description
        - Each #environment# must have id, name, and detailed description without characters
        - Each #scene# must reference a valid #environment#
        - Each #clip# must have id, characters list, and description
        2. Rules compliance:
        - Must have 2-4 #environments#
        - Must have 1-3 #characters# (A single clip is allowed to have 0 characters)
        - Animal #characters# should have natural appearances (no anthropomorphic physical traits)
        - #Environment# descriptions must not include #characters# or certain part of plot elements
        - No plant #characters# allowed
        - New #scene# only when #environment# changes (a #environment# can be used in different #scenes#)
            If Scene 1 and Scene 3 both take place in the Environment 1, the script should still pass the rule.
        - #Clips# should include visual actions, camera movements, and natural dialogue
        - Any single #clip# has no more than 2 #characters#.
        - There exits at leat a transitions clip in the whole script (refering entering or leaving some #environment#)
            eg1."Jimmy stood up with a look of disappointment, turned back and walked away, the camera zoomed out." eg2."Jimmy walks from the left side of the picture and greets the fox." 
            Even if only one scene contains a transition clip, while the rest of the scenes do not, the script should still pass the rule.

        First, provide a validation report with:
        - PASS/FAIL overall status
        - Specific issues found for each criterion
        - Suggestions for improvement if issues are found
        Do not include any JSON content in the above response.

        Then, if there are any issues (FAIL), provide a corrected JSON version of the script that fixes all the issues. 
        Format the corrected script as a valid JSON object directly. No additional text or markdown formatting. (```json ... ```)
        If there isn't any issue, no JSON version of the script is needed.
        """
        self.rules_for_short = """
        Rules:
        1. Include 1 or 2 #environments# and 1 or 2 or 3 #characters# and detailed #description# related. Totally 5-8 clips should be Included.
        2. Animals can serve as #characters#. They may possess the ability to talk or exhibit anthropomorphic behaviors, but their #character descriptions# should not include any anthropomorphic appearance elements. Instead, the descriptions should be based on their natural appearances, providing sufficient detail to enable accurate illustrations of them. Plants cannot serve as #characters#.
        3. Any #environment# should be the background where part of the storyboard takes place. Do not include any character in the #environment description# part, nor any characteristics connected to the plot. #Environment description# Should be suitable for use as a storyboard background, easy to depict, and not overshadow the main story.
        4. New #scene# only when #environment# changes, and any #environment# can be used in different #scenes#.
        5. Each #scene# contains sequential clips with:
        - #Environment# reference
        - Present #characters#
        - Visual actions (camera moves/transitions)
        - Natural dialogue (when speaking)
        6. At most 2 #characters# can be included in a #clip#.
        7. Consider adding some #clips# of entering/exiting the #environment# at the beginning/end of the #scene#. eg1."Jimmy stood up with a look of disappointment, turned back and walked away, the camera zoomed out." eg2."Jimmy walks from the left side of the picture and greets the fox." Note that the previous/next #environment# should not be included in the #clip description#.
        8. Use present tense action descriptions.
        9. Return the content in json format directly without any guide words or markdown.
        """
        self.check_for_short = """
        Evaluate this animation script against the following criteria:

        1. Schema validation:
        - Verify the script contains #characters#, #environments#, and #script#
        - Each #character# must have id, name, and detailed description
        - Each #environment# must have id, name, and detailed description without characters
        - Each #scene# must reference a valid #environment#
        - Each #clip# must have id, characters list, and description
        2. Rules compliance:
        - Must have 1-2 #environments#
        - Must have 1-3 #characters# (A single clip is allowed to have 0 characters)
        - Must have 5-8 #clips# in total
        - Animal #characters# should have natural appearances (no anthropomorphic physical traits)
        - #Environment# descriptions must not include #characters# or certain part of plot elements
        - No plant #characters# allowed
        - New #scene# only when #environment# changes
        - #Clips# should include visual actions, camera movements, and natural dialogue
        - Any single #clip# has no more than 2 #characters#.
        - There exits at leat a transitions clip in the whole script (refering entering or leaving some #environment#)
            eg1."Jimmy stood up with a look of disappointment, turned back and walked away, the camera zoomed out." eg2."Jimmy walks from the left side of the picture and greets the fox." 
            Even if only one scene contains a transition clip, while the rest of the scenes do not, the script should still pass the rule.

        First, provide a validation report with:
        - PASS/FAIL overall status
        - Specific issues found for each criterion
        - Suggestions for improvement if issues are found
        Do not include any JSON content in the above response.

        Then, if there are any issues (FAIL), provide a corrected JSON version of the script that fixes all the issues. 
        Format the corrected script as a valid JSON object directly. No additional text or markdown formatting. (```json ... ```)
        If there isn't any issue, no JSON version of the script is needed.
        """

    def generate_long_script(self):
        # Format the prompt combining story, schema and rules
        prompt = f"""
        ###
        {self.story}
        ###
        Create a structured animation script in JSON format based on the provided story. Follow this exact schema:
        ###
        {self.schema}
        ###
        The following rules must be adhered to:
        ###
        {self.rules_for_long}
        ###
        """
        generated_script = self.gemini_api.generate_from_text(prompt=prompt, model="gemini-2.0-flash")
        #generated_script = self.gemini_r1_api.generate_from_text(prompt=prompt, model="gemini-2.0-flash-thinking-exp")
        return generated_script
    
    def generate_short_script(self):
        # Format the prompt combining story, schema and rules
        prompt = f"""
        ###
        {self.story}
        ###
        Create a structured animation script in JSON format based on the provided story. Follow this exact schema:
        ###
        {self.schema}
        ###
        The following rules must be adhered to:
        ###
        {self.rules_for_short}
        ###
        """
        generated_script = self.gemini_api.generate_from_text(prompt=prompt, model="gemini-2.0-flash")
        return generated_script

    def validate_long_script(self, script_json):
        # Convert the script back to a JSON string for the API
        script_str = json.dumps(script_json, ensure_ascii=False)
        # Send the script and check prompt to Gemini for validation
        validation_prompt = f"""
        ###
        Animation Script:
        {script_str}
        ###
        {self.check_for_long}
        """
        #validation_result = self.gemini_api.generate_from_text(prompt=validation_prompt, model="gemini-2.0-flash")
        validation_result = self.deepseek_api.generate_from_text(prompt=validation_prompt, model="deepseek-reasoner")
        return validation_result
    
    def validate_short_script(self, script_json):
        script_str = json.dumps(script_json, ensure_ascii=False)
        validation_prompt = f"""
        ###
        Animation Script:
        {script_str}
        ###
        {self.check_for_short}
        """
        validation_result = self.deepseek_api.generate_from_text(prompt=validation_prompt, model="deepseek-reasoner")
        return validation_result
        
    def extract_script(self, text_result):
        cleaned_script = text_result
        
        # First try to extract content between markdown code blocks if present
        if "```" in cleaned_script:
            # Extract content between markdown code blocks
            blocks = cleaned_script.split("```")
            # The content should be in odd-indexed blocks
            for i in range(1, len(blocks), 2):
                # If it's a json block or doesn't have a language specifier
                if i < len(blocks) and (blocks[i].startswith("json\n") or not blocks[i].strip().split("\n")[0].isalpha()):
                    # Remove "json" language specifier if present
                    potential_json = blocks[i]
                    if potential_json.startswith("json\n"):
                        potential_json = potential_json[5:]  # Skip "json\n"
                    
                    # Try to parse this block
                    try:
                        return json.loads(potential_json.strip())
                    except json.JSONDecodeError:
                        # Continue to the next block if this one fails
                        continue
        
        # If no valid JSON found in code blocks, try cleaning the whole text
        cleaned_script = cleaned_script.replace("```json", "").replace("```", "").strip()
        
        # Try to extract JSON from the cleaned text
        try:
            # Try to parse the entire cleaned script as JSON
            return json.loads(cleaned_script)
        except json.JSONDecodeError:
            # If that fails, try to find JSON object within the text
            json_start = cleaned_script.find('{')
            json_end = cleaned_script.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                try:
                    json_str = cleaned_script[json_start:json_end]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
        
        return None

    def generate_long(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Generate the script
        script = self.generate_long_script()
        
        try:
            # Extract the script
            script_json = self.extract_script(script)
            if script_json is None:
                raise json.JSONDecodeError("Failed to parse script JSON", script, 0)
            
            tmpfile_dir = os.path.join(output_dir, "tmpfiles")
            os.makedirs(tmpfile_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "tmpfiles/script_ori.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(script_json, f, indent=2, ensure_ascii=False)
            
            print(f"Script successfully saved to {output_path}")
             
            # Loop until validation passes or max attempts reached
            current_script = script_json
            max_attempts = 5
            attempt = 1
            
            while attempt <= max_attempts:
                print(f"\nValidation attempt {attempt}/{max_attempts}...")
                validation_result = self.validate_long_script(current_script)
                
                # Save validation report for this attempt
                validation_path = os.path.join(output_dir, f"tmpfiles/validation_report_{attempt}.txt")
                with open(validation_path, "w", encoding="utf-8") as f:
                    f.write(validation_result)
                print(f"Validation report saved to {validation_path}")

                # Check if validation passed
                if "FAIL" not in validation_result:
                    print("Validation PASSED!")
                    break
                else:
                    print("Validation FAILED!")
                
                # If validation failed and we haven't reached max attempts, try to correct
                if attempt < max_attempts:
                    corrected_script = self.extract_script(validation_result)
                    if corrected_script:
                        # Save the corrected script
                        corrected_path = os.path.join(output_dir, f"tmpfiles/script_corrected_{attempt}.json")
                        with open(corrected_path, "w", encoding="utf-8") as f:
                            json.dump(corrected_script, f, indent=2, ensure_ascii=False)
                        print(f"Corrected script (attempt {attempt}) saved to {corrected_path}")
                        # Use corrected script for next validation attempt
                        current_script = corrected_script
                    else:
                        print(f"No valid corrected script found in validation attempt {attempt}")
                        # Save the raw validation output for debugging
                        raw_path = os.path.join(output_dir, f"tmpfiles/validation_raw_{attempt}.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(validation_result)
                        print(f"Raw validation output saved to {raw_path} for debugging")
                        # Break if we can't extract a valid corrected script
                        break
                attempt += 1
            
            # Save the final script
            final_path = os.path.join(output_dir, "script.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(current_script, f, indent=2, ensure_ascii=False)
            print(f"Final script saved to {final_path}")
            
            return current_script
            
        except json.JSONDecodeError as e:
            print(f"Error: Generated script is not valid JSON: {e}")
            # Save the raw output for debugging
            raw_path = os.path.join(output_dir, "tmpfiles/script_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(script)
            print(f"Raw output saved to {raw_path} for debugging")
            return None
    
    def generate_short(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        script = self.generate_short_script()
        try:
            script_json = self.extract_script(script)
            if script_json is None:
                raise json.JSONDecodeError("Failed to parse script JSON", script, 0)
            
            tmpfile_dir = os.path.join(output_dir, "tmpfiles")
            os.makedirs(tmpfile_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "tmpfiles/script_ori.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(script_json, f, indent=2, ensure_ascii=False)
            print(f"Script successfully saved to {output_path}")

            current_script = script_json
            max_attempts = 5
            attempt = 1
            
            while attempt <= max_attempts:
                print(f"\nValidation attempt {attempt}/{max_attempts}...")
                validation_result = self.validate_short_script(current_script)
                
                validation_path = os.path.join(output_dir, f"tmpfiles/validation_report_{attempt}.txt")
                with open(validation_path, "w", encoding="utf-8") as f:
                    f.write(validation_result)
                print(f"Validation report saved to {validation_path}")

                if "FAIL" not in validation_result:
                    print("Validation PASSED!")
                    break
                else:
                    print("Validation FAILED!")
                
                if attempt < max_attempts:
                    corrected_script = self.extract_script(validation_result)
                    if corrected_script:
                        corrected_path = os.path.join(output_dir, f"tmpfiles/script_corrected_{attempt}.json")
                        with open(corrected_path, "w", encoding="utf-8") as f:
                            json.dump(corrected_script, f, indent=2, ensure_ascii=False)
                        print(f"Corrected script (attempt {attempt}) saved to {corrected_path}")
                        current_script = corrected_script
                    else:
                        print(f"No valid corrected script found in validation attempt {attempt}")
                        raw_path = os.path.join(output_dir, f"tmpfiles/validation_raw_{attempt}.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(validation_result)
                        print(f"Raw validation output saved to {raw_path} for debugging")
                        break
                attempt += 1
            
            final_path = os.path.join(output_dir, "script.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(current_script, f, indent=2, ensure_ascii=False)
            print(f"Final script saved to {final_path}")
            
            return current_script
            
        except json.JSONDecodeError as e:
            print(f"Error: Generated script is not valid JSON: {e}")
            # Save the raw output for debugging
            raw_path = os.path.join(output_dir, "tmpfiles/script_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(script)
            print(f"Raw output saved to {raw_path} for debugging")
            return None


# # Example usage:
# if __name__ == "__main__":
#     story = """
#     One dawn, a mysterious seedling sprouted in the desert. The Little Prince nurtured it, whispering, "You'll be beautiful." Days later, a rose bloomed, glowing under the sun. "You're stunning!" he said. "I'm not ready!" she replied, fussing about the cold nights. He shielded her with a glass dome, but her endless demands broke his heart.
#     Fleeing westward, he met a wise fox. They played. Suddenly, the prince stumbled upon a wall of roses, their holographic petals shimmering in the wind. "Who are you?" he asked, stunned. "We are roses," they replied in unison. His heart sank. "My rose told me she was the only one," he muttered, confused and frustrated. The fox taught him: "Your rose is unique because of the time you shared."
#     Guided by this truth, the prince returned. Under frost-kissed glass, his rose shimmered weakly. "She's mine," he murmured, touching the dome. In her imperfect beauty, he finally saw love's essence—not perfection, but the bond forged through care.
#     """
    
#     Story2ScriptGenerator = Story2Script(story)
#     Story2ScriptGenerator.generate_long()
