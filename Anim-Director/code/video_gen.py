import os
import re
import json
import subprocess
from PIA.pia_api import PiaAPI

class VideoGenerator:
    def __init__(self, story_list, result_file):
        self.story_list = story_list
        self.result_file = result_file
        self.results = {}


    def save_results(self):
        with open(self.result_file, 'w') as file:
            json.dump(self.results, file, indent=4)

    
    def get_scene_no(self, i):
        segment = i % 2 + 1
        scene = i // 2 + 1  
        return f"Scene {scene} Segment {segment}"


    def extract_scene_segment(self, file_name):
        match = re.match(r'Scene_(\d+)_Segment_(\d+)', file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None


    def convert_gif_to_mp4(self, gif_file, mp4_file):
        ffmpeg_command = [
            'ffmpeg',
            '-i', gif_file,
            '-movflags', 'faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            mp4_file
        ]
        subprocess.run(ffmpeg_command, check=True)


    def concat_videos(self, id):
        video_dir = f'code/result/video/{id}'
        gif_files = [f for f in os.listdir(video_dir) if f.endswith('.gif')]
        gif_files.sort(key=lambda x: self.extract_scene_segment(x))
        mp4_files = []
        
        for gif_file in gif_files:
            gif_path = os.path.join(video_dir, gif_file)
            mp4_file = gif_file.replace('.gif', '.mp4')
            mp4_path = os.path.join(video_dir, mp4_file)
            self.convert_gif_to_mp4(gif_path, mp4_path)
            mp4_files.append(mp4_file)

        file_list_path = os.path.join(video_dir, 'file_list.txt')
        with open(file_list_path, 'w') as file_list:
            for mp4_file in mp4_files:
                file_list.write(f"file '{mp4_file}'\n")
        
        output_path = os.path.join(video_dir, f'{id}.mp4')
        
        ffmpeg_command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            output_path
        ]
        
        subprocess.run(ffmpeg_command, check=True)


    def Image2Video(self, id): # ---generate prompts for the drawing tool (eg. MidJourney)
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        if(self.results[str(id)] == {}):
            return
        segment_num = self.results[str(id)]['scene2image']['segment_num']
        image2video_data = {}
        
        print(f"******************No.{id} Image2Video Begin******************")
        for i in range(segment_num):
            sceneno = self.get_scene_no(i)
            prompt = self.results[str(id)]['scene2image'][sceneno]['scene']
            prompt = prompt[:10]
            prompts = [[prompt]]
            input_path = f'code/result/image/sd3/{id}/Scenes'
            input_name = sceneno.replace(' ', '_')
            save_path = os.path.join(f'code/result/video/{id}', sceneno.replace(' ', '_'))

            pia_api = PiaAPI(prompts, input_path, input_name, save_path)
            pia_api.generate()

            image2video_data[sceneno] = {'prompt': prompt, "input_path": input_path, "input_name": save_path, "input_name": save_path}
        
        self.concat_videos(id)
        self.results[str(id)]['image2video'] = image2video_data   
        with open(self.result_file, 'r') as file:
            self.results = json.load(file)
        print(f"******************No.{id} Image2Video Completed******************")


    def main(self):
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        for idNo in story_list:
            self.Image2Video(idNo)
            self.save_results()


if __name__ == '__main__':
    story_list = [32, 34, 54] # modify as you want
    result_file = 'code/result/script.json'

    generator = VideoGenerator(story_list, result_file)
    generator.main()