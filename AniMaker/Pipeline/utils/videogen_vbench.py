import os
import sys
sys.path.append("Tools/Wan2.1")
import re
import gc
import cv2
import json
import math
import time
import torch
import shutil
from Wan import WanI2V14B
from Tools.RealESRGAN.RealESRGAN import RealESRGAN
from Tools.gemini_api import GeminiAPI
from Tools.deepseek_api import DeepSeekAPI
from Tools.eval import Evaluator
from Benchmark.VBench.VBench import VBenchEvaluator

class VideoGenerator:
    def __init__(self, script_data, story_dir, generation_data, deepseek_key, gemini_keys, gemini_proxy_setting):
        self.script_data = script_data
        self.story_dir = story_dir
        self.generation_data = generation_data
        self.deepseek_key = deepseek_key
        self.gemini_keys = gemini_keys
        self.gemini_proxy_setting = gemini_proxy_setting
        self.res_folder = story_dir
        self.output_dir = os.path.join(self.res_folder, 'videos')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize models and APIs
        self.i2V_model = WanI2V14B()
        self.deepseek_api = DeepSeekAPI(api_key=self.deepseek_key, proxy=self.gemini_proxy_setting)
        self.gemini_api = GeminiAPI(api_keys=self.gemini_keys, proxy=self.gemini_proxy_setting)
        self.vbench_evaluator = VBenchEvaluator()

        # Initialize state
        self.chosen_path = []
        self.processed_clips = {}
        self._load_progress()

        # Number of candidate videos to generate per clip
        self.num_candidates = 5
        
        self.prompt_template = f""" 
        The prompt must align with the visual and contextual details of the first frame and the provided description and follow the structure below:
        Prompt Structure: The overall structure should follow Subject + Motion + slightly Camera Movement + ,3D animation.
        Subject: Use descriptive phrases like "the boy in green", "the fox" etc., instead of names like "Jimmy", "Lucy" to refer to the characters.
        Motion: Provide detailed descriptions of the motion characteristics, including amplitude, speed, and the effect of the movement, e.g., "violently swaying," "slowly moving," "shattering the glass."
        Camera Movement: Incorporate techniques such as slightly zooming in, slightly zooming out, or following the subject only if necessary.

        The structure can be either:
        (Subject A + Motion A + Subject B + Motion B (+ slightly Camera Movement only if necessary) ...)
        or
        Subject A and Subject B + Motion X (+ slightly Camera Movement only if necessary) + ,3D animation.

        For example:
        The boy in green stands up, turns back, and then fetches the blue watering can on the ground behind him, Camera following the boy, 3D animation.

        Note:
        1. When describing motion and interactions with objects, ensure detailed descriptions—including color, location, and other relevant details—are provided, such as "fetch the blue watering can on the ground behind him.
        2. Don’t add anything that can’t be shown on the video screen. For example, the content of spoken words should not be included in the prompt.
        3. Assign more common and simple actions to characters, and avoid overly complex or delicate actions or objects.
        4. Do not abuse the Camera Movement. If necessary, add "slightly" before "zooming in" or "zooming out".
        4. Directly give the simplified prompt without any guide words or markdown.
        """

    def _load_progress(self):
        """Load progress from generation_data."""
        if not self.generation_data.get("video_generation"):
            self.generation_data["video_generation"] = []
        else:
            for clip_data in self.generation_data["video_generation"]:
                scene_id = clip_data.get("scene_id")
                clip_id = clip_data.get("clip_id")
                if scene_id is not None and clip_id is not None:
                    if scene_id not in self.processed_clips:
                        self.processed_clips[scene_id] = set()
                    self.processed_clips[scene_id].add(clip_id)
                    # If this is the latest processed clip, restore chosen path
                    if clip_data.get("chosen_path"):
                        self.chosen_path = clip_data["chosen_path"]
                        print(f"Restored chosen path from scene {scene_id}, clip {clip_id}")
            print(f"Found {sum(len(clips) for clips in self.processed_clips.values())} already processed clips")

    def _find_first_unprocessed_clip(self):
        for scene_idx, scene in enumerate(self.script_data['script']):
            scene_id = scene_idx + 1  # Convert to 1-indexed
            for clip in scene['clips']:
                clip_id = clip['id']
                if scene_id not in self.processed_clips or clip_id not in self.processed_clips[scene_id]:
                    return scene_id, clip_id, clip
        return None, None, None
    
    def _get_character_info(self, clip):
        """Get character information for a specific clip."""
        characters_info = []
        character_dict = {}

        for char_id in clip['characters']:
            character = next((c for c in self.script_data['characters'] if str(c['id']) == char_id), None)
            if not character:
                print(f"Warning: Character with ID {char_id} not found in character list")
                continue

            character_name = character['name']
            character_description = character['description']
            formatted_name = character_name.replace(' ', '_')
            char_img_path = os.path.join(self.res_folder, f'imgs/characters/{formatted_name}/img.jpg')

            characters_info.append({
                "id": character['id'],
                "name": character_name,
                "description": character_description,
                "image_path": char_img_path if os.path.exists(char_img_path) else None
            })

            if os.path.exists(char_img_path):
                character_dict[character_name] = char_img_path

        return characters_info, character_dict
    
    def _get_environment_info(self, scene_id):
        """Get environment information for a specific scene."""
        scene = self.script_data['script'][scene_id - 1]  # Convert to 0-indexed
        env_id = scene['environment']
        environment = next((e for e in self.script_data['environments'] if e['id'] == env_id), None)

        if not environment:
            print(f"Warning: Environment with ID {env_id} not found for scene {scene_id}")
            return None

        env_name = environment['name']
        env_description = environment['description']
        formatted_env_name = env_name.replace(' ', '_')
        env_img_path = os.path.join(self.res_folder, f'imgs/environments/{formatted_env_name}/{formatted_env_name}.png')

        return {
            "id": env_id,
            "name": env_name,
            "description": env_description,
            "image_path": env_img_path if os.path.exists(env_img_path) else None
        }

    def _find_previous_clip(self, current_scene_id, current_clip_id):
        # Find the scene that contains the current clip
        current_scene = None
        current_scene_idx = None
        for i, scene in enumerate(self.script_data['script']):
            if i + 1 == current_scene_id:  # Convert to 1-indexed
                current_scene = scene
                current_scene_idx = i
                break
        # Find the current clip's index within its scene
        current_clip_idx = None
        for i, clip in enumerate(current_scene['clips']):
            if clip['id'] == current_clip_id:
                current_clip_idx = i
                break
        # Check if there's a previous clip in the same scene
        if current_clip_idx > 0:
            prev_clip = current_scene['clips'][current_clip_idx - 1]
            return current_scene_id, prev_clip['id'], prev_clip
        # If not, check if there's a previous scene with clips
        if current_scene_idx > 0:
            prev_scene = self.script_data['script'][current_scene_idx - 1]
            if prev_scene['clips']:
                prev_clip = prev_scene['clips'][-1]  # Get the last clip from previous scene
                return current_scene_idx, prev_clip['id'], prev_clip
        return None, None, None

    def _find_next_clip(self, current_scene_id, current_clip_id):
        # Find the scene that contains the current clip
        current_scene = None
        current_scene_idx = None
        for i, scene in enumerate(self.script_data['script']):
            if i + 1 == current_scene_id:  # Convert to 1-indexed
                current_scene = scene
                current_scene_idx = i
                break
        # Find the current clip's index within its scene
        current_clip_idx = None
        for i, clip in enumerate(current_scene['clips']):
            if clip['id'] == current_clip_id:
                current_clip_idx = i
                break
        # Check if there's a next clip in the same scene
        if current_clip_idx < len(current_scene['clips']) - 1:
            next_clip = current_scene['clips'][current_clip_idx + 1]
            return current_scene_id, next_clip['id'], next_clip
        # If not, check if there's a next scene with clips
        if current_scene_idx < len(self.script_data['script']) - 1:
            next_scene = self.script_data['script'][current_scene_idx + 1]
            if next_scene['clips']:
                next_clip = next_scene['clips'][0]
                return current_scene_idx + 2, next_clip['id'], next_clip  # +2 for 1-indexed
        return None, None, None
    
    def _determine_input_image_path(self, scene_id, clip_id):
        """Determine the input image path for a clip."""
        scene_name = f"scene_{scene_id}"
        clip_folder = os.path.join(self.output_dir, scene_name, f"clip_{clip_id}")
        os.makedirs(clip_folder, exist_ok=True)
        ref_frame_path = os.path.join(clip_folder, "ref_frame.jpg")

        scene_image_found = False
        input_image_path = None

        # Check for scene generation image
        for scene_gen in self.generation_data.get("scene_generation", []):
            if (scene_gen.get("scene_name") == scene_name and
                scene_gen.get("clip_id") == clip_id and
                scene_gen.get("generations")):
                gen_img_path = scene_gen["generations"][0]["output_image"]
                if os.path.exists(gen_img_path):
                    img = cv2.imread(gen_img_path)
                    cv2.imwrite(ref_frame_path, img)
                    input_image_path = ref_frame_path
                    scene_image_found = True
                    print(f"Using scene generation image for scene {scene_id}, clip {clip_id}")
                    break

        # Check for reusable videos from previous clip
        reusable_videos = []
        if self.chosen_path:
            prev_scene_id, prev_clip_id, _ = self._find_previous_clip(scene_id, clip_id)
            if prev_scene_id and prev_clip_id:
                prev_clip_node = None
                for node in self.chosen_path:
                    if node["scene_id"] == prev_scene_id and node["clip_id"] == prev_clip_id:
                        prev_clip_node = node
                        break
                if prev_clip_node:
                    last_video_path = prev_clip_node["video_info"]["enhanced_video"]
                    if not os.path.exists(ref_frame_path) and os.path.exists(last_video_path):
                        cap = cv2.VideoCapture(last_video_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_count > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                            ret, frame = cap.read()
                            cap.release()
                            if ret:
                                cv2.imwrite(ref_frame_path, frame)
                                print(f"Saved last frame from previous clip (scene {prev_scene_id}, clip {prev_clip_id}) to {ref_frame_path}")
                                input_image_path = ref_frame_path
                            else:
                                print(f"Warning: Failed to extract last frame from {last_video_path}")
                        else:
                             print(f"Warning: Video {last_video_path} has 0 frames.")
                             cap.release()
                    elif os.path.exists(ref_frame_path):
                         input_image_path = ref_frame_path # Already exists, use it

        return input_image_path, scene_image_found, reusable_videos

    def _generate_params_from_prompt(self, prompt):
        """Generate mm_action, raft_amp, and sam_count from prompt using DeepSeek API"""
        instruction = f"""
        Analyze the following prompt for an animated scene and extract three pieces of information:

        1. Main action (mm_action): Extract the primary action being performed by the human character(if no human character exists, choose the main animal character instead) in the scene.
           Output format: Just the action verb (e.g., "walking", "talking", "touching")
           If no action is found, return "None"

        2. Motion speed (raft_amp): Determine if the motion in the scene is performed fastly or slowly.
           Output format: Just "fast" or "slow"

        3. Count of main objects (sam_count): Count one kind of significant object mentioned in the prompt(Only one kind).
           Output format: Number + object name (e.g., "1 boy"; "2 flowers")

        Prompt: {prompt}

        Output each answer on a separate line without any additional text or explanation.
        """
        response = self.deepseek_api.generate_from_text(instruction)
        lines = response.strip().split('\n')
        mm_action = lines[0].strip()
        raft_amp = lines[1].strip()
        sam_count = lines[2].strip()
        return mm_action, raft_amp, sam_count

    def _update_pipeline_json(self):
        """Helper to save the current state to the pipeline JSON file."""
        pipeline_json_path = os.path.join(self.story_dir, 'pipeline.json')
        with open(pipeline_json_path, 'w') as f:
            json.dump(self.generation_data, f, indent=4)

    def generate_videos(self):
        """Main loop to generate videos for all clips."""
        print("Starting or resuming video generation...")

        while True:
            current_scene_id, current_clip_id, current_clip = self._find_first_unprocessed_clip()
            if current_clip is None:
                print("All clips have been processed!")
                break

            scene_name = f"scene_{current_scene_id}"
            clip_folder = os.path.join(self.output_dir, scene_name, f"clip_{current_clip_id}")
            os.makedirs(clip_folder, exist_ok=True)
            print(f"\n{'='*30}\nProcessing video for scene {current_scene_id}, clip {current_clip_id}\n{'='*30}")

            characters_info, character_dict = self._get_character_info(current_clip)
            env_info = self._get_environment_info(current_scene_id)
            prev_scene_id, prev_clip_id, prev_clip = self._find_previous_clip(current_scene_id, current_clip_id)
            next_scene_id, next_clip_id, next_clip = self._find_next_clip(current_scene_id, current_clip_id)

            prev_clip_info = {"scene_id": prev_scene_id, "clip_id": prev_clip_id, "description": prev_clip['description']} if prev_clip else None
            next_clip_info = {"scene_id": next_scene_id, "clip_id": next_clip_id, "description": next_clip['description']} if next_clip else None

            input_image_path, scene_image_found, reusable_videos = self._determine_input_image_path(current_scene_id, current_clip_id)
            mm_action, raft_amp, sam_count = self._generate_params_from_prompt(current_clip['description'])

            clip_data = {
                "scene_id": current_scene_id,
                "clip_id": current_clip_id,
                "description": current_clip['description'],
                "characters": characters_info,
                "environment": env_info,
                "mm_action": mm_action,
                "raft_amp": raft_amp,
                "sam_count": sam_count,
                "previous_clip": prev_clip_info,
                "next_clip_info": next_clip_info,
                "candidates": [],
                "chosen_path": self.chosen_path # Store path leading to this clip
            }

            # Set number of candidates to generate
            num_candidates = self.num_candidates
            print(f"Generating {num_candidates} candidate videos")

            pre_video_path = self.chosen_path[-1]["video_info"]["output_video"] if self.chosen_path else None

            # STEP 1: Generate all candidate videos
            candidates = []
            for video_idx in range(num_candidates):
                shift = 5.0
                guide_scale = 7.5
                seed = 42 + video_idx  # Different seed for each candidate

                # Generate prompt
                formatted_prompt = self.deepseek_api.generate_from_text(
                    current_clip['description'] + '\n' + self.prompt_template
                )

                prompt = formatted_prompt
                output_video_path = os.path.join(
                    clip_folder,
                    f"{scene_name}_clip_{current_clip_id}_candidate_{video_idx+1}.mp4"
                )
                print(f"Generated prompt for candidate video {video_idx+1}:\n{formatted_prompt}")

                # Generate the video
                video = self.i2V_model.generate(
                    prompt=prompt,
                    image_path=input_image_path,
                    save_file=output_video_path,
                    shift=shift,
                    guide_scale=guide_scale,
                    seed=seed
                )
                print(f"Generated candidate video {video_idx+1} saved to {output_video_path}")

                # Enhance the video
                enhanced_folder = os.path.join(clip_folder, "enhanced")
                os.makedirs(enhanced_folder, exist_ok=True)
                enhanced_video_name = os.path.basename(output_video_path).replace(".mp4", "_outx4.mp4")
                enhanced_video_path = os.path.join(enhanced_folder, enhanced_video_name)

                print(f"Enhancing video with RealESRGAN...")
                try:
                    upscaler = RealESRGAN()
                    upscaler.enhance_video(input_path=output_video_path, output_path=enhanced_folder)
                    del upscaler
                except Exception as e:
                    print(f"Error during RealESRGAN enhancement: {e}")
                    enhanced_video_path = output_video_path # Fallback to original if enhancement fails
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Store candidate video information without evaluation yet
                candidate = {
                    "candidate_id": video_idx + 1,
                    "prompt": prompt,
                    "image_path": input_image_path,
                    "output_video": output_video_path,
                    "enhanced_video": enhanced_video_path,
                    "parameters": {"shift": shift, "guide_scale": guide_scale, "seed": seed}
                }
                candidates.append(candidate)

            # STEP 2: Run VBench evaluation on all candidate videos together
            print(f"\nEvaluating all {len(candidates)} candidate videos with VBench...")
            vbench_results = self.vbench_evaluator.evaluate(
                videos_path=clip_folder
            )
            
            # Update candidate information with evaluation results
            video_rankings = vbench_results.get("combined_ranks", {})
            best_video_path = vbench_results.get("best_overall_video")
            best_candidate_match = re.search(r'_candidate_(\d+)\.mp4$', best_video_path)
            best_candidate_id = int(best_candidate_match.group(1))
                
            print(f"Selected best candidate {best_video_path}")

            # Store best candidate in the chosen path
            chosen_node_info = {
                "scene_id": current_scene_id,
                "clip_id": current_clip_id,
                "candidate_id": best_candidate_id,
                "video_info": candidates[best_candidate_id - 1],  # -1 for 0-indexed
            }
            self.chosen_path.append(chosen_node_info)
            
            # Update clip data with candidates and chosen path
            clip_data["candidates"] = candidates
            clip_data["chosen_path"] = self.chosen_path
            clip_data["best_candidate_id"] = best_candidate_id

            # Save clip data to main generation data
            existing_clip_index = next((index for (index, d) in enumerate(self.generation_data["video_generation"])
                                        if d.get("scene_id") == current_scene_id and d.get("clip_id") == current_clip_id), None)
            if existing_clip_index is not None:
                self.generation_data["video_generation"][existing_clip_index] = clip_data
            else:
                self.generation_data["video_generation"].append(clip_data)

            self._update_pipeline_json()

            # Mark as processed
            if current_scene_id not in self.processed_clips:
                self.processed_clips[current_scene_id] = set()
            self.processed_clips[current_scene_id].add(current_clip_id)

        # Clean up models
        self.cleanup()
        print("Video generation completed!")
        return self.generation_data

    def process_chosen_path(self):
        """Processes the final chosen path after all videos are generated."""
        if not self.chosen_path:
            print("No chosen path available to process.")
            return

        print("\nProcessing final chosen path...")
        chosen_dir = os.path.join(self.res_folder, 'videos/chosen')
        os.makedirs(chosen_dir, exist_ok=True)
        final_chosen_path_info = []

        for node_info in self.chosen_path:
            scene_id = node_info["scene_id"]
            clip_id = node_info["clip_id"]
            video_info = node_info["video_info"]
            source_video = video_info.get("enhanced_video") or video_info.get("output_video") # Fallback

            if not source_video or not os.path.exists(source_video):
                print(f"Warning: Source video not found for scene {scene_id}, clip {clip_id} (Path: {source_video}). Skipping.")
                continue

            video_filename = f"scene_{scene_id}_clip_{clip_id}.mp4"
            dest_video = os.path.join(chosen_dir, video_filename)

            try:
                shutil.copy2(source_video, dest_video)
                print(f"Saved chosen path video for scene {scene_id}, clip {clip_id} to {dest_video}")
                final_chosen_path_info.append({
                    "scene_id": scene_id,
                    "clip_id": clip_id,
                    "candidate_id": node_info["candidate_id"],
                    "source_video": source_video,
                    "chosen_video_path": dest_video,
                    "score": video_info.get("vbench_score")
                })
            except Exception as e:
                print(f"Error saving chosen path video for scene {scene_id}, clip {clip_id}: {str(e)}")

        self.generation_data["final_chosen_path"] = final_chosen_path_info
        self._update_pipeline_json()
        print("Chosen path processing completed!")

    def cleanup(self):
        """Release resources."""
        if hasattr(self, 'i2V_model') and self.i2V_model:
            del self.i2V_model
        torch.cuda.empty_cache()
        gc.collect()
        print("VideoGenerator resources cleaned up.")
