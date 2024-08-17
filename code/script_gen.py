import os
import re
import json
from tool.gpt import GPT

class ScriptGenerator:
    def __init__(self, story_file, story_list, prompt_file, result_file, gpt_organization, gpt_api_key, proxy, scene_number):
        self.story_file = story_file
        self.story_list = story_list
        self.prompt_file = prompt_file
        self.result_file = result_file
        self.gpt_organization = gpt_organization
        self.gpt_api_key = gpt_api_key
        self.proxy = proxy
        self.scene_number = scene_number
        self.results = {}
        self.variables = self.load_variables()
        self.story = self.load_story()
        self.setup_proxy()


    def setup_proxy(self):
        try:
            if self.proxy:
                os.environ['HTTP_PROXY'] = self.proxy
                os.environ['HTTPS_PROXY'] = self.proxy
                http_proxy_set = os.environ.get('HTTP_PROXY') == self.proxy
                https_proxy_set = os.environ.get('HTTPS_PROXY') == self.proxy
                if http_proxy_set and https_proxy_set:
                    print("Proxy setup successful.")
                else:
                    print("Proxy setup failed. Please check the proxy settings.")
            else:
                print("No proxy provided.")
        except Exception as e:
            print(f"An error occurred while setting up the proxy: {e}")


    def load_variables(self):
        variables = {}
        with open(self.prompt_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if 'f"' in value:
                    variables[key] = value
                else:
                    variables[key] = eval(value)
        return variables
    

    def format_prompt(self, prompt):
        if prompt.startswith('f"') and prompt.endswith('"'):
            return prompt[2:-1]
        return prompt


    def load_story(self):
        with open(self.story_file, 'r') as file:
            return json.load(file)


    def save_results(self):
        with open(self.result_file, 'w') as file:
            json.dump(self.results, file, indent=4)


    def Story2Scene(self, id):
        if os.path.getsize(self.result_file) > 0:
            with open(self.result_file, 'r') as file:
                self.results = json.load(file)
        for item in self.story:
            if item['id'] == id:
                story = item['story']
        
        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        story2scene_data = {}  
        story2scene_data['story'] = story
        words_number = self.scene_number * 30
        story2scene_step1_prompt = self.format_prompt(self.variables['story2scene_step1_prompt'].format(words_num=words_number))
        story2scene_step3_prompt_3 = self.format_prompt(self.variables['story2scene_step3_prompt_3'].format(scene_num=self.scene_number))
        story2scene_step3_prompt_5 = self.format_prompt(self.variables['story2scene_step3_prompt_5'].format(scene_num=self.scene_number))

        # story2scene step1 ---expand the story
        print(f"******************No.{id} Story2Scene Step1 Begin******************")
        story2scene_step1_question = story2scene_step1_prompt + "'''" + story + "'''"
        print(f"Story2Scene Step1 Question:\n{story2scene_step1_question}\n")
        story2scene_step1_answer = gpt_api.query(story2scene_step1_question)
        story2scene_step1_answer = '\n'.join([line for line in story2scene_step1_answer.split('\n') if line.strip() != ''])
        story2scene_data['step1_answer'] = story2scene_step1_answer
        print(f"Story2Scene Step1 Answer:\n{story2scene_step1_answer}\n")

        # story2scene step2 ---design characters and settings
        print(f"******************No.{id} Story2Scene Step2 Begin******************")
        characters_dict, settings_dict = {}, {}
        characters_list, settings_list = [], []
        story2scene_step2_question = self.variables['story2scene_step2_prompt_1'] + "'''" + story2scene_step1_answer + "'''" + self.variables['story2scene_step2_prompt_2']
        print(f"Story2Scene Step2 Question:\n{story2scene_step2_question}\n")
        while(characters_dict == {} or settings_dict == {}):
            story2scene_step2_answer = gpt_api.query(story2scene_step2_question)
            story2scene_data['step2_answer'] = story2scene_step2_answer
            print(f"Story2Scene Step2 Answer:\n{story2scene_step2_answer}\n")
            try:
                characters_list = story2scene_step2_answer.split("Characters list:")[1].split("Settings list:")[0].strip().strip("[]").split(", ")
                settings_list = story2scene_step2_answer.split("Settings list:")[1].split("Part 1.")[0].strip().strip("[]").split(", ")
                if not characters_list or not settings_list:
                    raise ValueError("Extraction of characters or settings names failed.")
                characters_dict = {name: "" for name in characters_list}
                settings_dict = {name: "" for name in settings_list}
                characters_description = story2scene_step2_answer.split("Part 1. Characters:")[1].split("Part 2.")[0].strip()
                for character in characters_list:
                    start = characters_description.find(character + ":") + len(character) + 2
                    end = characters_description.find("\n", start)
                    description = characters_description[start:end].strip()
                    if not description:
                        raise ValueError(f"Description extraction failed for character: {character}")
                    characters_dict[character] = description
                # Extract descriptions for settings
                settings_description = story2scene_step2_answer.split("Part 2. Settings:")[1].strip()
                for setting in settings_list:
                    start = settings_description.find(setting + ":") + len(setting) + 2
                    end = settings_description.find("\n", start)
                    description = settings_description[start:end].strip()
                    if not description:
                        raise ValueError(f"Description extraction failed for setting: {setting}")
                    settings_dict[setting] = description
                # Ensure all keys have been filled
                if not all(characters_dict.values()) or not all(settings_dict.values()):
                    raise ValueError("Not all descriptions were successfully extracted.")
            except Exception as e:
                characters_dict, settings_dict = {}, {}
        
        final_flag = 0
        retry_times = 0
        while(final_flag == 0 and retry_times < 5):
            # story2scene step3 ---design the script
            print(f"******************No.{id} Story2Scene Step3 Begin******************")
            story2scene_step3_question = self.variables['story2scene_step3_prompt_1'] + "'''" + story2scene_step1_answer + "'''\n" + self.variables['story2scene_step3_prompt_2'] + 'Characters:\n' + str(characters_dict) + '\nSettings:\n' + str(settings_dict) + '\n' + story2scene_step3_prompt_3 + str(characters_list) + '. ' + self.variables['story2scene_step3_prompt_4'] + str(settings_list) + '. ' + story2scene_step3_prompt_5
            print(f"Story2Scene Step3 Question:\n{story2scene_step3_question}\n")
            story2scene_step3_answer = gpt_api.query(story2scene_step3_question)
            story2scene_step3_answer = '\n'.join([line for line in story2scene_step3_answer.split('\n') if line.strip() != ''])
            story2scene_data['step3_answer'] = story2scene_step3_answer
            print(f"Story2Scene Step3 Answer:\n{story2scene_step3_answer}\n")

            # story2scene step4 ---check the script
            print(f"******************No.{id} Story2Scene Step4 Begin******************")  
            characters_str = '\n'.join([f"{key}: {value}" for key, value in characters_dict.items()])
            settings_str = '\n'.join([f"{key}: {value}" for key, value in settings_dict.items()])
            script = 'Characters:\n' + characters_str + '\nSettings:\n' + settings_str + '\nScenes:\n' + story2scene_step3_answer 
            story2scene_step4_answer = ""
            story2scene_step4_redo = 0
            while ("No problem found".lower() not in story2scene_step4_answer.lower()) and (story2scene_step4_redo <= 3):
                if "New Version of Complete Script".lower() in story2scene_step4_answer.lower():
                    print(f"******************No.{id} Story2Scene Step4 Redo******************")
                    matches = re.search("New Version of Complete Script.*?Characters:(.*)", story2scene_step4_answer, re.DOTALL)
                    if matches:
                        script = 'Characters:\n' + matches.group(1).strip().strip("'").strip()
                    else:
                        break
                elif story2scene_step4_redo != 0:
                    print("No New Version of Complete Script")
                    break
                story2scene_step4_question = self.variables['story2scene_step4_prompt_1'] + "'''" + script + "'''" + self.variables['story2scene_step4_prompt_2']
                print(f"Story2Scene Step4 Question:\n{story2scene_step4_question}\n")
                story2scene_step4_answer = gpt_api.query(story2scene_step4_question)
                key = f'step4_answer_redo{story2scene_step4_redo}' if story2scene_step4_redo > 0 else 'step4_answer'
                story2scene_data[key] = story2scene_step4_answer
                print(f"Story2Scene Step4 Answer:\n{story2scene_step4_answer}\n")
                story2scene_step4_redo += 1

            lines = script.split('\n')
            for i, line in enumerate(lines):
                line = line.rstrip()
                if line and line[-1] not in '.!?,:;':
                    lines[i] = line + '.'
            script = '\n'.join(lines)

            # Check if the characters and settings in the script are consistent
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, script)
            character_flag = 1 
            setting_flag = 1
            for i in range(0, len(matches), 2):
                if i < len(matches):
                    names = [name.strip() for name in matches[i].split(',')]
                    if not all(name in characters_list for name in names):
                        character_flag = 0
            for i in range(1, len(matches), 2):
                if i < len(matches):
                    places = [place.strip() for place in matches[i].split(',')]
                    if not all(place in settings_list for place in places):
                        setting_flag = 0
            # Check if enough scenes have been generated
            pattern = r'Scene (\d+)'
            numbers = re.findall(pattern, script)
            numbers = [int(num) for num in numbers]
            current_scene_num = max(numbers) if numbers else 0
            
            if(character_flag == 1 and setting_flag == 1 and current_scene_num >= self.scene_number):
                final_flag = 1
            retry_times += 1
        
        if(retry_times == 5):
            self.results[str(id)] = {}
            self.save_results()
            print(f"Script Generation for Story No.{id} Failed!\n")
            return
        
        story2scene_data['final_answer'] = script
        print(f"Story2Scene Final Answer:\n{script}\n")
        if str(id) not in self.results:
            self.results[str(id)] = {}
        self.results[str(id)]['story2scene'] = story2scene_data
        print(f"******************No.{id} Story2Scene Completed******************")


    def Scene2Segment(self, id):
        if(self.results[str(id)] == {}):
            return
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        story2scene_answer = self.results[str(id)]['story2scene']['final_answer']
        scene_numbers = [int(num) for num in re.findall(r"Scene (\d+)", story2scene_answer)]
        scene_num = max(scene_numbers) if scene_numbers else 0
        segment_num = scene_num * 2
        self.variables['scene_num'] = scene_num
        self.variables['segment_num'] = segment_num

        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        scene2segment_data = {}
        scene2segment_data['scene'] = story2scene_answer
        scene2segment_step1_prompt_2 = self.format_prompt(self.variables['scene2segment_step1_prompt_2'].format(scene_num=scene_num, segment_num=segment_num))
        scene2segment_step2_prompt_2 = self.format_prompt(self.variables['scene2segment_step2_prompt_2'].format(segment_num=segment_num))
        scene2segment_step3_prompt_1 = self.format_prompt(self.variables['scene2segment_step3_prompt_1'].format(segment_num=segment_num))
        scene2segment_step3_prompt_2 = self.format_prompt(self.variables['scene2segment_step3_prompt_2'].format(segment_num=segment_num))
        scene2segment_step4_prompt = self.format_prompt(self.variables['scene2segment_step4_prompt'].format(segment_num=segment_num))

        # scene2segment step1 ---divide each scene into 2 segments for the script
        print(f"******************No.{id} Scene2Segment Step1 Begin******************")
        scene2segment_step1_question = self.variables['scene2segment_step1_prompt_1'] + "'''" + story2scene_answer + "'''" + scene2segment_step1_prompt_2
        print(f"Scene2Segment Step1 Question:\n{scene2segment_step1_question}\n")
        scene2segment_step1_answer = gpt_api.query(scene2segment_step1_question)
        scene2segment_step1_answer = '\n'.join([line for line in scene2segment_step1_answer.split('\n') if line.strip() != ''])
        scene2segment_data['step1_answer'] = scene2segment_step1_answer
        print(f"Scene2Segment Step1 Answer:\n{scene2segment_step1_answer}\n")

        part1_part2_content = re.search(r'(Characters.*?Scenes)', story2scene_answer, re.DOTALL) 
        part1_part2_content = part1_part2_content.group(0)
        part1_part2_content = re.sub(r'^(Characters|Settings|Scenes)$', r'\1:', part1_part2_content, flags=re.MULTILINE) # Add a ':' after "Characters", "Settings" or "Scenes" on a separate line
        scene2segment_step1_answer = part1_part2_content + "\n" + scene2segment_step1_answer
        scene2segment_step1_answer = re.sub(r'^\s*$\n', '', scene2segment_step1_answer, flags=re.MULTILINE)
        
        final_flag = 0
        retry_times = 0
        while(final_flag == 0 and retry_times < 5):
            current_answer = scene2segment_step1_answer
            # scene2segment step2 ---check the script
            print(f"******************No.{id} Scene2Segment Step2 Begin******************")
            scene2segment_step2_answer = ""
            scene2segment_step2_redo = 0
            while ("No problem found".lower() not in scene2segment_step2_answer.lower()) and (scene2segment_step2_redo <= 3):
                if "New Version of Complete Script:".lower() in scene2segment_step2_answer.lower():
                    print(f"******************No.{id} Scene2Segment Step2 Redo******************")
                    matches = re.search("New Version of Complete Script.*?Characters:(.*)", scene2segment_step2_answer, re.DOTALL)
                    if matches:
                        current_answer = 'Characters:\n' + matches.group(1).strip().strip("'").strip()
                elif scene2segment_step2_redo != 0:
                    print("No New Version of Complete Script")
                    break
                scene2segment_step2_question = self.variables['scene2segment_step2_prompt_1'] + "'''" + current_answer + "'''" + scene2segment_step2_prompt_2
                print(f"Scene2Segment Step2 Question:\n{scene2segment_step2_question}\n")
                scene2segment_step2_answer = gpt_api.query(scene2segment_step2_question)
                key = f'step2_answer_redo{scene2segment_step2_redo}' if scene2segment_step2_redo > 0 else 'step2_answer'
                scene2segment_data[key] = scene2segment_step2_answer
                print(f"Scene2Segment Step2 Answer:\n{scene2segment_step2_answer}\n")
                scene2segment_step2_redo += 1

            print(f"Scene2Segment Intermediate Answer:\n{current_answer}")
            
            # scene2segment step3 ---specify the character layout of each segment
            print(f"******************No.{id} Scene2Segment Step3 Begin******************")
            story2scene_step2_answer = self.results[str(id)]['story2scene']['step2_answer']
            characters_list = story2scene_step2_answer.split("Characters list:")[1].split("Settings list:")[0].strip().strip("[]").split(", ")
            settings_list = story2scene_step2_answer.split("Settings list:")[1].split("Part 1.")[0].strip().strip("[]").split(", ")
            index = current_answer.find("Scenes:\n")
            scenes_content = current_answer[(index+8):]
            
            scene2segment_step3_question = "Characters list: " + str(characters_list) + "\nScene description:\n" + scenes_content + "\n" + scene2segment_step3_prompt_1 + str(characters_list) + scene2segment_step3_prompt_2
            print(f"Scene2Segment Step3 Question:\n{scene2segment_step3_question}\n")
            scene2segment_step3_answer = gpt_api.query(scene2segment_step3_question)
            scene2segment_step3_answer = '\n'.join([line for line in scene2segment_step3_answer.split('\n') if line.strip() != ''])
            scene2segment_data['step3_answer'] = scene2segment_step3_answer
            print(f"Scene2Segment Step3 Answer:\n{scene2segment_step3_answer}\n")
            pattern = r'\[(.*?)\]'
            step1_matches = re.findall(pattern, current_answer)
            step2_matches = re.findall(pattern, scene2segment_step3_answer)
            for i in range(0, len(step1_matches), 2):
                if i < len(step1_matches):
                    escaped_match = re.escape(step1_matches[i]) 
                    current_answer = re.sub(f'\\[{escaped_match}\\]', f'[{step2_matches[i//2]}]', current_answer, count=1)
            print(current_answer)
            
            # scene2segment step4 ---double-check the characters in each segment to ensure they are consistent with the character setting
            print(f"******************No.{id} Scene2Segment Step4 Begin******************")
            scene2segment_step4_question = f"The full description of the {segment_num} segments is:\n" + scenes_content + "\n" + scene2segment_step4_prompt 
            print(f"Scene2Segment Step4 Question:\n{scene2segment_step4_question}\n")
            scene2segment_step4_answer = gpt_api.query(scene2segment_step4_question)
            scene2segment_step4_answer = '\n'.join([line for line in scene2segment_step4_answer.split('\n') if line.strip() != ''])
            scene2segment_data['step4_answer'] = scene2segment_step4_answer
            print(f"Scene2Segment Step4 Answer:\n{scene2segment_step4_answer}\n")
            index = current_answer.find("Scenes:\n")
            part1_part2_content = current_answer[:(index+8)]
            current_answer = part1_part2_content + scene2segment_step4_answer

            scene2segment_answer = current_answer # Step2-step4 is the inspection and replacement step, directly correcting the result of step1
            lines = scene2segment_answer.split('\n')
            for i, line in enumerate(lines):
                line = line.rstrip()
                if line and line[-1] not in '.!?,:;':
                    lines[i] = line + '.'
            scene2segment_answer = '\n'.join(lines)

            # Check if the characters and settings in the script are consistent
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, scene2segment_answer)
            character_flag = 1 
            setting_flag = 1
            for i in range(0, len(matches), 2):
                if i < len(matches):
                    names = [name.strip() for name in matches[i].split(',')]
                    if not all(name in characters_list for name in names):
                        character_flag = 0
            for i in range(1, len(matches), 2):
                if i < len(matches):
                    places = [place.strip() for place in matches[i].split(',')]
                    if not all(place in settings_list for place in places):
                        setting_flag = 0
            # Check if enough scenes have been generated
            pattern = r'Scene (\d+)'
            numbers = re.findall(pattern, scene2segment_answer)
            numbers = [int(num) for num in numbers]
            current_scene_num = max(numbers) if numbers else 0
            
            if(character_flag == 1 and setting_flag == 1 and current_scene_num >= self.scene_number):
                final_flag = 1
            retry_times += 1
        
        if(retry_times == 5):
            self.results[str(id)] = {}
            self.save_results()
            print(f"Script Generation for Story No.{id} Failed!\n")
            return

        scene2segment_data['final_answer'] = scene2segment_answer
        print(f"Scene2Segment Final Answer:\n{scene2segment_answer}\n")
        self.results[str(id)]['scene2segment'] = scene2segment_data   
        print(f"******************No.{id} Scene2Segment Completed******************")


    def Segment2Prompt(self, id): # ---generate prompts for the drawing tool (eg. MidJourney)
        if(self.results[str(id)] == {}):
            return
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        scene2segment_answer = self.results[str(id)]['scene2segment']['final_answer']
        story2scene_answer = self.results[str(id)]['story2scene']['final_answer']
        scene_numbers = [int(num) for num in re.findall(r"Scene (\d+)", story2scene_answer)]
        scene_num = max(scene_numbers) if scene_numbers else 0
        segment_num = scene_num * 2

        gpt_api = GPT(organization=self.gpt_organization, api_key=self.gpt_api_key)
        segment2prompt_data = {}
        segment2prompt_data['segment'] = scene2segment_answer
        segment2prompt_prompt_3 = self.format_prompt(self.variables['segment2prompt_prompt_3'].format(segment_num=segment_num))

        final_flag = 0
        retry_times = 0
        while(final_flag == 0 and retry_times < 5):
            print(f"******************No.{id} Segment2Prompt Begin******************")
            story2scene_step2_answer = self.results[str(id)]['story2scene']['step2_answer']
            characters_list = story2scene_step2_answer.split("Characters list:")[1].split("Settings list:")[0].strip().strip("[]").split(", ")
            settings_list = story2scene_step2_answer.split("Settings list:")[1].split("Part 1.")[0].strip().strip("[]").split(", ")
            segment2prompt_question = self.variables['segment2prompt_prompt_1'] + "'''" + scene2segment_answer + "'''" + self.variables['segment2prompt_prompt_2'] + "Characters list: " + str(characters_list) + segment2prompt_prompt_3
            print(f"Segment2Prompt Question:\n{segment2prompt_question}\n")
            segment2prompt_answer = gpt_api.query(segment2prompt_question)
            segment2prompt_answer = '\n'.join([line for line in segment2prompt_answer.split('\n') if line.strip() != ''])
            segment2prompt_data['answer'] = segment2prompt_answer
            print(f"Segment2Prompt Answer:\n{segment2prompt_answer}\n")

            index = scene2segment_answer.find("Scenes:\n")
            part1_part2_content = scene2segment_answer[:(index+8)]
            segment2prompt_answer = part1_part2_content + segment2prompt_answer
            lines = segment2prompt_answer.split('\n')
            for i, line in enumerate(lines):
                line = line.rstrip()
                if line and line[-1] not in '.!?,:;':
                    lines[i] = line + '.'
            segment2prompt_answer = '\n'.join(lines)

            # Check if the characters and settings in the script are consistent
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, segment2prompt_answer)
            character_flag = 1 
            setting_flag = 1
            for i in range(0, len(matches), 2):
                if i < len(matches):
                    names = [name.strip() for name in matches[i].split(',')]
                    if not all(name in characters_list for name in names):
                        character_flag = 0
            for i in range(1, len(matches), 2):
                if i < len(matches):
                    places = [place.strip() for place in matches[i].split(',')]
                    if not all(place in settings_list for place in places):
                        setting_flag = 0
            # Check if enough scenes have been generated
            pattern = r'Scene (\d+)'
            numbers = re.findall(pattern, segment2prompt_answer)
            numbers = [int(num) for num in numbers]
            current_scene_num = max(numbers) if numbers else 0
            
            if(character_flag == 1 and setting_flag == 1 and current_scene_num >= self.scene_number):
                final_flag = 1
            retry_times += 1
        
        if(retry_times == 5):
            self.results[str(id)] = {}
            self.save_results()
            print(f"Script Generation for Story No.{id} Failed!\n")
            return

        segment2prompt_data['final_answer'] = segment2prompt_answer
        print(f"Segment2Prompt Final Answer:\n{segment2prompt_answer}\n")

        self.results[str(id)]['segment2prompt'] = segment2prompt_data   
        print(f"******************No.{id} Segment2Prompt Completed******************")


    def main(self):
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        for idNo in story_list:
            self.Story2Scene(idNo)
            self.save_results()
            self.Scene2Segment(idNo)
            self.save_results()
            self.Segment2Prompt(idNo)
            self.save_results()


if __name__ == '__main__':
    story_file = './dataset/TinyStoriesV2-Chosen.json'
    story_list = [32, 34, 54] # modify as you want
    prompt_file = './code/prompt/script_gen.txt'
    result_file = './code/result/script.json'
    gpt_organization = "your gpt_organization"
    gpt_api_key = "your gpt_api_key"
    proxy = 'your proxy'
    scene_number = 10

    generator = ScriptGenerator(story_file, story_list, prompt_file, result_file, gpt_organization, gpt_api_key, proxy, scene_number)
    generator.main()