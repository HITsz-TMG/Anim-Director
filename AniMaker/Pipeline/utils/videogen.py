import os
import sys
sys.path.append("Tools/Wan2.1")
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

class VideoGenerator:
    def __init__(self, script_data, story_dir, generation_data, deepseek_key, gemini_keys, gemini_proxy_setting):
        self.script_data = script_data
        self.story_dir = story_dir
        self.generation_data = generation_data
        self.deepseek_key = deepseek_key
        self.gemini_keys = gemini_keys
        self.gemini_proxy_setting = gemini_proxy_setting
        self.res_folder = story_dir # Use story_dir as the base for results
        self.output_dir = os.path.join(self.res_folder, 'videos')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize models and APIs
        self.i2V_model = WanI2V14B()
        self.deepseek_api = DeepSeekAPI(api_key=self.deepseek_key, proxy=self.gemini_proxy_setting)
        self.gemini_api = GeminiAPI(api_keys=self.gemini_keys, proxy=self.gemini_proxy_setting)

        # Initialize state
        self.chosen_path = []
        self.processed_clips = {}
        self._load_progress()
        
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
        5. Directly give the simplified prompt without any guide words or markdown.
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

        pregenerated_videos = []
        if self.chosen_path:
            last_chosen_node_info = self.chosen_path[-1]
            # Find the full node data in generation_data for the previous clip
            prev_clip_data = next((cd for cd in self.generation_data.get("video_generation", [])
                                   if cd.get("scene_id") == last_chosen_node_info["scene_id"] and
                                      cd.get("clip_id") == last_chosen_node_info["clip_id"]), None)

            if prev_clip_data:
                # Find the selected node from the previous clip
                selected_prev_node = next((n for n in prev_clip_data.get("uct_nodes", [])
                                           if n.get("node_id") == last_chosen_node_info["node_id"]), None)

                if selected_prev_node:
                    # Find its children that were generated for the current clip
                    for child_id in selected_prev_node.get("children", []):
                        # Search across all clip_data for the child node
                        child_node = None
                        for cd in self.generation_data.get("video_generation", []):
                             found_node = next((n for n in cd.get("uct_nodes", []) if n.get("node_id") == child_id), None)
                             if found_node:
                                 child_node = found_node
                                 break

                        if child_node and child_node.get("for_next_clip"):
                            next_clip_info = child_node.get("video_info", {}).get("next_clip_info")
                            if next_clip_info and (next_clip_info.get("scene_id") == scene_id and
                                                    next_clip_info.get("clip_id") == clip_id):
                                pregenerated_videos.append(child_node)
                                print(f"Found pregenerated video for scene {scene_id}, clip {clip_id} from previous UCT exploration (Node {child_node.get('node_id')})")


        prev_scene_id, prev_clip_id, _ = self._find_previous_clip(scene_id, clip_id)

        if not scene_image_found and self.chosen_path and prev_scene_id and prev_clip_id:
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

        return input_image_path, scene_image_found, pregenerated_videos


    def _generate_params_from_prompt(self, prompt):
            """Generate mm_action, raft_amp, and sam_count from prompt using DeepSeek API"""
            instruction = f"""
            Analyze the following prompt for an animated scene and extract three pieces of information:

            1. Main action (mm_action): Extract the primary action being performed by the human character(if no human character exists, choose the main animal character instead) in the scene.
               Output format: Just the action verb (e.g., "walking", "talking", "touching")
               If no action is found, return "None"

            2. Motion speed (raft_amp): Determine if the motion in the scene is performed fast or slowly.
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
        # print(f"Updated generation information in {pipeline_json_path}") # Reduce verbosity
  
    def generate_videos(self):
        """Main loop to generate videos for all clips using UCT."""
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

            input_image_path, scene_image_found, pregenerated_videos = self._determine_input_image_path(current_scene_id, current_clip_id)
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
                "uct_nodes": [],
                "chosen_path": self.chosen_path.copy() # Store path leading to this clip
            }

            ds_prompt_template = f"""
            I am an Animation Director utilizing text-to-video models (such as Wan 2.1) to create high-quality animations. My task is to generate a detailed and precise prompt for the model to produce animations based on the scene description. The description for the given image is as follows:
            ### {current_clip['description']} ###
            """
            gemini_prompt_template = f"""
            I am an Animation Director utilizing text+image-to-video models (such as Wan 2.1) to create high-quality animations. My task is to generate a detailed and precise prompt for the model to produce animations based on the first frame and the corresponding scene description. The first frame has already been uploaded, and the description for the given image is as follows:
            ### {current_clip['description']} ###
            """

            w1 = 3  # Number of initial videos to generate
            w2 = 3  # Number of UCT iterations
            alpha = 1  # UCT hyperparameter
            
            uct_nodes = [] # Nodes for the current clip processing
            # STEP 1: INITIAL EXPLORATION
            reused_nodes = []
            videos_to_generate = w1

            if pregenerated_videos:
                print(f"Found {len(pregenerated_videos)} pregenerated videos for this clip.")
                for idx, pregenerated_node in enumerate(pregenerated_videos):
                    # Ensure the pregenerated node has necessary info
                    if pregenerated_node and pregenerated_node.get("video_info"):
                         # Create a new node structure for the current clip's UCT process
                        reused_node = {
                            "node_id": len(uct_nodes) + len(reused_nodes) + 1,
                            "rank": 0,
                            "parent": None,
                            "children": [],
                            "child_count": 0,
                            "child_scores": [],
                            "uct_value": 0,
                            "is_reused": True,
                            "original_node_id": pregenerated_node.get("node_id"),
                            "video_info": pregenerated_node.get("video_info") # Copy video info
                        }
                        # Ensure evaluation data exists before adding
                        if reused_node["video_info"].get("evaluation", {}).get("total_score") is not None:
                            reused_nodes.append(reused_node)
                            print(f"Reusing pregenerated node {pregenerated_node.get('node_id')} as new node {reused_node['node_id']}")
                        else:
                            print(f"Warning: Pregenerated node {pregenerated_node.get('node_id')} lacks evaluation data, skipping reuse.")
                    else:
                        print(f"Warning: Invalid pregenerated node data found: {pregenerated_node}")

                videos_to_generate = max(0, w1 - len(reused_nodes))
                print(f"Reused {len(reused_nodes)} valid pregenerated videos, need to generate {videos_to_generate} more.")

            print(f"Initial exploration: Generating {videos_to_generate} new videos + {len(reused_nodes)} reused videos")

            pre_video_path = self.chosen_path[-1]["video_info"]["output_video"] if self.chosen_path else None

            initial_generations = []
            initial_generations.extend(reused_nodes) # Add valid reused nodes first

            for video_idx in range(videos_to_generate):
                shift = 5.0
                guide_scale = 7.5
                seed = 42 + video_idx + len(reused_nodes) # Adjust seed based on reused count

                # Generate prompt based on index
                if video_idx < (w1 - 1): # Prioritize Gemini for first few
                    api_name = "Gemini"
                    formatted_prompt = self.gemini_api.generate_multimodal(
                        prompt=gemini_prompt_template + '\n' + self.prompt_template,
                        image_paths=[input_image_path]
                    ) if input_image_path and os.path.exists(input_image_path) else self.gemini_api.generate_from_text(
                        gemini_prompt_template + '\n' + self.prompt_template
                    )
                else:
                    api_name = "Deepseek"
                    formatted_prompt = self.deepseek_api.generate_from_text(
                        ds_prompt_template + '\n' + self.prompt_template
                    )

                prompt = formatted_prompt
                output_video_path = os.path.join(
                    clip_folder,
                    f"{scene_name}_clip_{current_clip_id}_initial_{video_idx+1}.mp4"
                )
                print(f"Generated prompt for initial video {video_idx+1} using {api_name}:\n{formatted_prompt}")

                # Generate the video
                video = self.i2V_model.generate(
                    prompt=prompt,
                    image_path=input_image_path,
                    save_file=output_video_path,
                    shift=shift,
                    guide_scale=guide_scale,
                    seed=seed
                )
                print(f"Generated initial video {video_idx+1} saved to {output_video_path}")

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

                # Evaluate the video
                print(f"Evaluating initial video {video_idx+1}...")
                evaluator = Evaluator()
                pre_results = {}
                pre_simplified = {}
                if pre_video_path and os.path.exists(pre_video_path):
                    pre_video_continuous = not scene_image_found
                    print(f"Evaluating continuity with previous video (continuous={pre_video_continuous})")
                    try:
                        pre_results = evaluator.evaluate_pre_continuity(
                            video_path=output_video_path,
                            pre_video_path=pre_video_path,
                            character_dict=character_dict,
                            pre_video_continuous=pre_video_continuous
                        )
                        for key, value in pre_results.items():
                            if key.endswith('_value'): pre_simplified[key.replace('_value', '')] = value
                            elif key.startswith('DS(DreamSim)_Pre'): pre_simplified[key] = value
                    except Exception as e:
                        print(f"Error during pre-continuity evaluation: {e}")

                simplified_scores, detailed_results = {}, {}
                try:
                    simplified_scores, detailed_results = evaluator.evaluate(
                        description = current_clip['description'],
                        image_path=input_image_path,
                        video_path=output_video_path,
                        mm_action=mm_action, raft_amp=raft_amp, sam_count=sam_count,
                        character_dict=character_dict
                    )
                except Exception as e:
                    print(f"Error during main evaluation: {e}")
                    simplified_scores = {} # Ensure it's a dict even on error

                combined_scores = simplified_scores.copy()
                combined_scores.update(pre_simplified)

                total_score, normalized_scores = evaluator.calculate_total_score(combined_scores)
                print(f"Initial video {video_idx+1} score: {total_score:.2f}/100")
                # print(f"Normalized Scores:\n{normalized_scores}") # Reduce verbosity

                evaluator.release_memory()
                del evaluator
                torch.cuda.empty_cache()
                time.sleep(2) # Shorter sleep

                # Extract last frame
                last_frame_path = os.path.join(clip_folder, f"last_frame_initial_{video_idx+1}.jpg")
                ret_frame = False
                if os.path.exists(enhanced_video_path):
                    cap = cv2.VideoCapture(enhanced_video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                        ret_frame, frame = cap.read()
                        if ret_frame: cv2.imwrite(last_frame_path, frame)
                    cap.release()

                node = {
                    "node_id": len(uct_nodes) + len(initial_generations) + 1,
                    "rank": 0, "parent": None, "children": [], "child_count": 0,
                    "child_scores": [], "uct_value": 0, "is_reused": False,
                    "video_info": {
                        "video_index": len(reused_nodes) + video_idx + 1, "prompt": prompt,
                        "image_path": input_image_path, "output_video": output_video_path,
                        "enhanced_video": enhanced_video_path, "last_frame": last_frame_path if ret_frame else None,
                        "parameters": {"shift": shift, "guide_scale": guide_scale, "seed": seed},
                        "evaluation": {
                            "simplified_scores": simplified_scores, "pre_continuity": pre_results,
                            "normalized_scores": {k: float(v) for k, v in normalized_scores.items()},
                            "total_score": float(total_score)
                        }
                    }
                }
                initial_generations.append(node)

            # Rank initial videos
            ranked_nodes = sorted(
                initial_generations,
                key=lambda n: n["video_info"]["evaluation"]["total_score"],
                reverse=True
            )

            # Assign ranks and calculate initial UCT values
            for rank, node in enumerate(ranked_nodes):
                node["rank"] = rank + 1
                node["uct_value"] = 2.0 / (node["rank"] + 1) + math.sqrt(2.0 / (node["child_count"] + 1)) * alpha
                uct_nodes.append(node) # Add to the main list for this clip
                print(f"Initial node {node['node_id']} - Rank: {node['rank']}, Score: {node['video_info']['evaluation']['total_score']:.2f}, UCT: {node['uct_value']:.4f}")

            clip_data["uct_nodes"].extend(uct_nodes) # Add initial nodes to clip_data

            # STEP 2: UCT EXPLORATION - Generate for the next clip
            if next_clip_info and next_clip:
                print(f"\nStarting UCT exploration to generate videos for next clip (scene {next_scene_id}, clip {next_clip_id})")
                next_clip_folder = os.path.join(self.output_dir, f"scene_{next_scene_id}", f"clip_{next_clip_id}")
                os.makedirs(next_clip_folder, exist_ok=True)
                next_ref_frame_path = os.path.join(next_clip_folder, "ref_frame.jpg")

                next_input_image_path = None
                next_scene_name = f"scene_{next_scene_id}"
                next_scene_image_found = False
                for scene_gen in self.generation_data.get("scene_generation", []):
                     if (scene_gen.get("scene_name") == next_scene_name and
                         scene_gen.get("clip_id") == next_clip_id and
                         scene_gen.get("generations")):
                        gen_img_path = scene_gen["generations"][0]["output_image"]
                        if os.path.exists(gen_img_path):
                            img = cv2.imread(gen_img_path)
                            cv2.imwrite(next_ref_frame_path, img)
                            next_input_image_path = next_ref_frame_path
                            next_scene_image_found = True
                            print(f"Using scene generation image for next clip (scene {next_scene_id}, clip {next_clip_id})")
                            break
                if not next_scene_image_found:
                    print(f"No scene generation image found for next clip. Will use last frames from parent nodes.")


                next_mm_action, next_raft_amp, next_sam_count = self._generate_params_from_prompt(next_clip['description'])
                next_ds_prompt_template = f"I am an Animation Director...\n### {next_clip['description']} ###" # Shortened
                next_gemini_prompt_template = f"I am an Animation Director...\n### {next_clip['description']} ###" # Shortened
                next_characters_info, next_character_dict = self._get_character_info(next_clip)

                for uct_iter in range(w2):
                    print(f"\nUCT Exploration Iteration {uct_iter+1}/{w2}")

                    # Select best initial node (parent=None) based on current UCT value
                    initial_nodes_for_selection = [n for n in uct_nodes if n["parent"] is None]
                    if not initial_nodes_for_selection:
                        print("Warning: No initial nodes found for UCT selection. Skipping iteration.")
                        continue

                    sorted_initial_nodes = sorted(initial_nodes_for_selection, key=lambda n: n["uct_value"], reverse=True)
                    best_node_to_expand = sorted_initial_nodes[0]

                    print(f"Selected node {best_node_to_expand['node_id']} for expansion with UCT: {best_node_to_expand['uct_value']:.4f}")

                    iteration_input_image = next_input_image_path # Use scene image if found
                    if not next_scene_image_found:
                        iteration_input_image = best_node_to_expand["video_info"]["last_frame"]
                        print(f"Using best node {best_node_to_expand['node_id']}'s last frame as input for next clip")

                    shift = 5.0
                    guide_scale = 7.5
                    seed = 42 + len(clip_data["uct_nodes"]) # Seed based on total nodes generated so far for this clip

                    api_name = "Gemini" if uct_iter < (w2 - 1) else "Deepseek"
                    if api_name == "Gemini" and iteration_input_image and os.path.exists(iteration_input_image):
                        formatted_prompt = self.gemini_api.generate_multimodal(
                            prompt=next_gemini_prompt_template + '\n' + self.prompt_template,
                            image_paths=[iteration_input_image]
                        )
                    else: # Fallback to Deepseek if no image or Gemini fails
                        api_name = "Deepseek" # Ensure correct API name if falling back
                        formatted_prompt = self.deepseek_api.generate_from_text(
                            next_ds_prompt_template + '\n' + self.prompt_template
                        )

                    prompt = formatted_prompt
                    output_video_path = os.path.join(
                        next_clip_folder,
                        f"scene_{next_scene_id}_clip_{next_clip_id}_pregen_from_scene{current_scene_id}_clip{current_clip_id}_node{best_node_to_expand['node_id']}_iter{uct_iter+1}.mp4"
                    )
                    print(f"Generated prompt for next clip using {api_name}:\n{formatted_prompt}")

                    # Generate the video for the next clip
                    video = self.i2V_model.generate(
                        prompt=prompt, image_path=iteration_input_image, save_file=output_video_path,
                        shift=shift, guide_scale=guide_scale, seed=seed
                    )
                    print(f"Generated child video for next clip saved to {output_video_path}")

                    # Enhance
                    enhanced_folder = os.path.join(next_clip_folder, "enhanced")
                    os.makedirs(enhanced_folder, exist_ok=True)
                    enhanced_video_name = os.path.basename(output_video_path).replace(".mp4", "_outx4.mp4")
                    enhanced_video_path = os.path.join(enhanced_folder, enhanced_video_name)
                    print(f"Enhancing next clip video...")
                    try:
                        upscaler = RealESRGAN()
                        upscaler.enhance_video(input_path=output_video_path, output_path=enhanced_folder)
                        del upscaler
                    except Exception as e:
                        print(f"Error during RealESRGAN enhancement: {e}")
                        enhanced_video_path = output_video_path # Fallback
                    finally:
                        torch.cuda.empty_cache(); gc.collect()

                    # Extract last frame
                    last_frame_path = os.path.join(
                        next_clip_folder,
                        f"last_frame_pregen_from_scene{current_scene_id}_clip{current_clip_id}_node{best_node_to_expand['node_id']}_iter{uct_iter+1}.jpg"
                    )
                    ret_frame = False
                    if os.path.exists(enhanced_video_path):
                        cap = cv2.VideoCapture(enhanced_video_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_count > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                            ret_frame, frame = cap.read()
                            if ret_frame: cv2.imwrite(last_frame_path, frame)
                        cap.release()

                    # Evaluate the generated child video
                    print(f"Evaluating next clip video (child) for UCT iteration {uct_iter+1}...")
                    evaluator = Evaluator()
                    pre_results = {}
                    pre_simplified = {}
                    parent_video_path = best_node_to_expand["video_info"]["output_video"]

                    if parent_video_path and os.path.exists(parent_video_path):
                        pre_video_continuous = True # Assumed continuous as it's generated from last frame
                        print(f"Evaluating pre-continuity for next clip with parent node {best_node_to_expand['node_id']}")
                        try:
                            pre_results = evaluator.evaluate_pre_continuity(
                                video_path=output_video_path, pre_video_path=parent_video_path,
                                character_dict=next_character_dict, pre_video_continuous=pre_video_continuous
                            )
                            for key, value in pre_results.items():
                                if key.endswith('_value'): pre_simplified[key.replace('_value', '')] = value
                                elif key.startswith('DS(DreamSim)_Pre'): pre_simplified[key] = value
                        except Exception as e:
                            print(f"Error during child pre-continuity evaluation: {e}")

                    simplified_scores, detailed_results = {}, {}
                    try:
                        simplified_scores, detailed_results = evaluator.evaluate(
                            description = next_clip['description'], # Use next clip's description
                            image_path=iteration_input_image, video_path=output_video_path,
                            mm_action=next_mm_action, raft_amp=next_raft_amp, sam_count=next_sam_count,
                            character_dict=next_character_dict
                        )
                    except Exception as e:
                        print(f"Error during child main evaluation: {e}")
                        simplified_scores = {}

                    combined_scores = simplified_scores.copy()
                    combined_scores.update(pre_simplified)

                    total_score, normalized_scores = evaluator.calculate_total_score(combined_scores)
                    print(f"Next clip video score: {total_score:.2f}/100")

                    evaluator.release_memory()
                    del evaluator
                    torch.cuda.empty_cache()
                    time.sleep(2)

                    # Create child node
                    child_node = {
                        "node_id": len(self.generation_data["video_generation"]) * 100 + len(clip_data["uct_nodes"]) - w1 + 1, # More unique ID
                        "rank": 0, "parent": best_node_to_expand["node_id"], "children": [],
                        "child_count": 0, "child_scores": [], "uct_value": 0,
                        "for_next_clip": True, "next_clip_info": next_clip_info,
                        "video_info": {
                            "video_index": len(clip_data["uct_nodes"]) + 1, "prompt": prompt,
                            "image_path": iteration_input_image, # Image used for this child
                            "output_video": output_video_path, "enhanced_video": enhanced_video_path,
                            "last_frame": last_frame_path if ret_frame else None,
                            "parameters": {"shift": shift, "guide_scale": guide_scale, "seed": seed},
                            "evaluation": {
                                "simplified_scores": simplified_scores, "pre_continuity": pre_results,
                                "combined_scores": combined_scores, # Store combined for potential reuse
                                "normalized_scores": {k: float(v) for k, v in normalized_scores.items()},
                                "total_score": float(total_score)
                            },
                            "for_next_clip": True,
                            "next_clip_info": { # Info for reuse identification
                                "scene_id": next_scene_id, "clip_id": next_clip_id,
                                "from_scene_id": current_scene_id, "from_clip_id": current_clip_id,
                                "from_node_id": best_node_to_expand["node_id"]
                            }
                        }
                    }

                    # Update parent node
                    best_node_to_expand["children"].append(child_node["node_id"])
                    best_node_to_expand["child_count"] += 1
                    best_node_to_expand["child_scores"].append(float(total_score))

                    # Update UCT value for the expanded parent node
                    best_node_to_expand["uct_value"] = (2.0/(best_node_to_expand["rank"] + 1) + math.sqrt(1.0/(best_node_to_expand["child_count"] + 1)) * alpha)
                    print(f"Updated UCT for initial node {best_node_to_expand['node_id']} with actual child score: {best_node_to_expand['uct_value']:.4f}")

                    # Add child node to the current clip's data (important for tracking)
                    clip_data["uct_nodes"].append(child_node)

            # STEP 3: SELECTION
            print("\nSelecting best node for chosen path...")
            best_combined_score = -1
            selected_node_for_path = None
            nodes_for_selection = [n for n in uct_nodes if n["parent"] is None] # Only consider initial nodes

            if not nodes_for_selection:
                print("Warning: No initial nodes available for final selection.")
            else:
                for node in nodes_for_selection:
                    node_score = node["video_info"]["evaluation"]["total_score"]
                    combined_score = node_score # Default score
                    # Initialize storage for pair scores within the node's evaluation data
                    if "pair_scores_with_children" not in node["video_info"]["evaluation"]:
                        node["video_info"]["evaluation"]["pair_scores_with_children"] = []

                    if node["child_count"] > 0:
                        # Evaluate post-continuity and calculate pair scores
                        pair_scores = []
                        node_video_path = node["video_info"]["output_video"]

                        for i, child_id in enumerate(node["children"]):
                            # Find the child node (it's stored in clip_data["uct_nodes"])
                            child_node = next((n for n in clip_data["uct_nodes"] if n["node_id"] == child_id), None)

                            if child_node and child_node.get("video_info"):
                                child_score = child_node["video_info"]["evaluation"]["total_score"]
                                child_video_path = child_node["video_info"]["output_video"]
                                post_simplified = {}

                                if node_video_path and os.path.exists(node_video_path) and \
                                   child_video_path and os.path.exists(child_video_path):
                                    post_video_continuous = True # Assumed continuous
                                    print(f"Evaluating post-continuity between node {node['node_id']} and child {child_id}")
                                    try:
                                        evaluator = Evaluator()
                                        post_results = evaluator.evaluate_post_continuity(
                                            video_path=node_video_path, post_video_path=child_video_path,
                                            character_dict=character_dict, # Use current clip's characters
                                            post_video_continuous=post_video_continuous
                                        )
                                        # Store post-continuity results
                                        if "post_continuity" not in node["video_info"]["evaluation"]:
                                            node["video_info"]["evaluation"]["post_continuity"] = {}
                                        node["video_info"]["evaluation"]["post_continuity"][str(child_id)] = post_results

                                        for key, value in post_results.items():
                                            if key.endswith('_value'): post_simplified[key.replace('_value', '')] = value
                                            elif key.startswith('DS(DreamSim)_Post'): post_simplified[key] = value # Corrected key check

                                        evaluator.release_memory(); del evaluator; torch.cuda.empty_cache()
                                    except Exception as e:
                                        print(f"Error during post-continuity evaluation: {e}")
                                        if 'evaluator' in locals(): evaluator.release_memory(); del evaluator; torch.cuda.empty_cache()

                                # Recalculate node score including post-continuity metrics for this specific child pair
                                after_post_scores = node["video_info"]["evaluation"]["simplified_scores"].copy()
                                after_post_scores.update(post_simplified) # Add post scores for this child

                                # Need an evaluator instance to recalculate
                                temp_evaluator = Evaluator()
                                after_post_total_score, _ = temp_evaluator.calculate_total_score(after_post_scores)
                                temp_evaluator.release_memory(); del temp_evaluator; torch.cuda.empty_cache()

                                pair_score = after_post_total_score + child_score # Pair score uses node score enhanced by post-continuity to *this* child
                                print(f"  Node {node['node_id']} (Post-enhanced: {after_post_total_score:.2f}) -> Child {child_id} ({child_score:.2f}) = Pair Score: {pair_score:.2f}")
                                pair_scores.append(pair_score)

                                # Store the pair score and child ID in the parent node
                                node["video_info"]["evaluation"]["pair_scores_with_children"].append({
                                    "child_node_id": child_id,
                                    "pair_score": float(pair_score),
                                    "parent_score": float(after_post_total_score),
                                    "child_score": float(child_score)
                                })

                        if pair_scores:
                            combined_score = sum(pair_scores) / len(pair_scores) # Average of pair scores
                            print(f"Node {node['node_id']} - Avg Pair Score with children: {combined_score:.2f}")
                        else:
                            print(f"Node {node['node_id']} - Score: {node_score:.2f} (no valid children found for post-eval)")

                    else:
                        print(f"Node {node['node_id']} - Score: {node_score:.2f} (no children generated)")

                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        selected_node_for_path = node

            if selected_node_for_path:
                print(f"Selected node {selected_node_for_path['node_id']} with combined score {best_combined_score:.2f} for chosen path")
                chosen_node_info = {
                    "scene_id": current_scene_id,
                    "clip_id": current_clip_id,
                    "node_id": selected_node_for_path["node_id"],
                    "video_info": selected_node_for_path["video_info"] # Store full info
                }
                self.chosen_path.append(chosen_node_info)
                clip_data["chosen_path"] = self.chosen_path.copy() # Update final path in clip data
                clip_data["best_node"] = selected_node_for_path["node_id"]
                print(f"Updated chosen path. Length: {len(self.chosen_path)}")
            else:
                print("Warning: No suitable node found for chosen path. Path not updated.")
                # Decide how to handle this - maybe pick the best initial node?
                if nodes_for_selection:
                    best_initial_node = sorted(nodes_for_selection, key=lambda n: n["video_info"]["evaluation"]["total_score"], reverse=True)[0]
                    print(f"Fallback: Selecting best initial node {best_initial_node['node_id']} based on score {best_initial_node['video_info']['evaluation']['total_score']:.2f}")
                    chosen_node_info = {
                    "scene_id": current_scene_id, "clip_id": current_clip_id,
                    "node_id": best_initial_node["node_id"], "video_info": best_initial_node["video_info"]
                    }
                    self.chosen_path.append(chosen_node_info)
                    clip_data["chosen_path"] = self.chosen_path.copy()
                    clip_data["best_node"] = best_initial_node["node_id"]
                else:
                    clip_data["best_node"] = None # Indicate failure

            # Save clip data to main generation data and update JSON
            # Check if clip_data for this scene/clip already exists and update it, otherwise append
            existing_clip_index = next((index for (index, d) in enumerate(self.generation_data["video_generation"])
                                        if d.get("scene_id") == current_scene_id and d.get("clip_id") == current_clip_id), None)
            if existing_clip_index is not None:
                self.generation_data["video_generation"][existing_clip_index] = clip_data
            else:
                self.generation_data["video_generation"].append(clip_data)

            self._update_pipeline_json() # Save progress after processing each clip

            # Mark as processed
            if current_scene_id not in self.processed_clips:
                self.processed_clips[current_scene_id] = set()
            self.processed_clips[current_scene_id].add(current_clip_id)

        # Clean up models after loop finishes
        self.cleanup()
        print("Video generation with UCT-based selection algorithm completed!")
        return self.generation_data # Return updated data

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
                    "node_id": node_info["node_id"],
                    "source_video": source_video,
                    "chosen_video_path": dest_video,
                    "score": video_info.get("evaluation", {}).get("total_score")
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
