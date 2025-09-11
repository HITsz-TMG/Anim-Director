import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
from Pipeline.utils.story2script import Story2Script
from Pipeline.utils.script2character import Script2Character
from Pipeline.utils.script2environment import Script2Environment
from Pipeline.utils.script2scene import Script2Scene
from Pipeline.utils.videogen import VideoGenerator
from Pipeline.utils.audiogen import CosyVoiceGenerator
from Pipeline.utils.finalvideogen import FinalVideoGenerator

deepseek_key = "your_deepseek_key_here"  # Replace with your actual DeepSeek key
gemini_keys = ['your_gemini_key_1_here', 'your_gemini_key_2_here', 'your_gemini_key_3_here']  # Replace with your actual Gemini keys
gemini_proxy_setting = 'your_proxy_here'  # Replace with your actual proxy setting
openai_key = "your_openai_key_here"  # Replace with your actual OpenAI key

# Load stories from TinyStoriesV2-Chosen.json
stories_file_path = 'Pipeline/TinyStoriesV2-Chosen.json'
with open(stories_file_path, 'r') as f:
    stories = json.load(f)

# Create the main results directory
results_base_dir = 'Pipeline/res'
os.makedirs(results_base_dir, exist_ok=True)

# Process counter to track how many stories have been processed
story_process_counter = 0

# Process each story
for story_entry in stories:
    story_id = story_entry["id"]
    story_text = story_entry["story"]

    # Increment the story counter at the beginning of each loop
    story_process_counter += 1

    print(f"\n{'='*50}")
    print(f"Processing Story ID: {story_id} (Story #{story_process_counter})")
    print(f"{'='*50}\n")

    # Create story-specific directory
    story_dir = os.path.join(results_base_dir, str(story_id))
    ### Waiting fot GPT api
    if not os.path.exists(story_dir):
        continue
    #os.makedirs(story_dir, exist_ok=True)

    # Define the pipeline JSON path for this story
    pipeline_json_path = os.path.join(story_dir, 'pipeline.json')

    # Initialize or load existing generation data
    generation_data = {
        "story_id": story_id,
        "story_text": story_text,
        "script_generation": {},
        "character_generation": [],
        "environment_generation": [],
        "scene_generation": [],
        "video_generation": [],
        "audio_generation": {},
        "final_chosen_path": []
    }

    # Check if pipeline.json already exists for this story
    if os.path.exists(pipeline_json_path):
        print(f"Loading existing pipeline data from {pipeline_json_path}")
        with open(pipeline_json_path, 'r') as f:
            generation_data = json.load(f)
    else:
        print(f"Starting new pipeline process for story {story_id}")

    # Function to update and save the pipeline JSON
    def update_pipeline_json():
        with open(pipeline_json_path, 'w') as f:
            json.dump(generation_data, f, indent=4)
        print(f"Updated generation information in {pipeline_json_path}")

    ## Script Generation
    if not generation_data.get("script_generation"):
        print(f"Starting script generation for story {story_id}...")

        # Track script generation input
        generation_data["script_generation"] = {
            "story_id": story_id,
            "story_input": story_text
        }

        Story2ScriptGenerator = Story2Script(story_text, deepseek_key, gemini_keys, gemini_proxy_setting)
        if story_process_counter <= 20:  # story_id = 170 开始为short
            print(f"Using generate_long() for story #{story_process_counter}")
            Story2ScriptGenerator.generate_long(output_dir=f"Pipeline/res/{story_id}")
        else:
            print(f"Using generate_short() for story #{story_process_counter}")
            Story2ScriptGenerator.generate_short(output_dir=f"Pipeline/res/{story_id}")

        ## Script Loading
        script_path = os.path.join(story_dir, 'script.json')
        # Copy the generated script to the story directory
        if os.path.exists(os.path.join(f"Pipeline/res/{story_id}", 'script.json')):
            with open(os.path.join(f"Pipeline/res/{story_id}", 'script.json'), 'r') as f:
                script_data = json.load(f)

            with open(script_path, 'w') as f:
                json.dump(script_data, f, indent=4)
        else:
            print("Warning: script.json not generated!")
            continue  # Skip to next story if script generation failed

        # Track script generation output
        generation_data["script_generation"]["script_output_path"] = script_path
        generation_data["script_generation"]["script_data"] = script_data

        # Update JSON after script generation
        update_pipeline_json()
    else:
        print(f"Script generation for story {story_id} already completed, loading data...")
        script_data = generation_data["script_generation"]["script_data"]

    ## Characters Image Generation
    if not generation_data.get("character_generation"):
        print(f"Starting character image generation for story {story_id}...")
        character_generator = Script2Character(script_data=script_data, story_dir=story_dir)
        character_results = character_generator.generate_characters()
        generation_data["character_generation"] = character_results
        update_pipeline_json()
    else:
        print(f"Character generation for story {story_id} already completed, skipping...")

    ## Environment Image Generation with Flux
    if not generation_data.get("environment_generation"):
        print(f"Starting environment image generation for story {story_id}...")
        environment_generator = Script2Environment(
            script_data=script_data,
            story_dir=story_dir,
            deepseek_api_key=deepseek_key,
            gemini_api_keys=gemini_keys,
            gemini_proxy=gemini_proxy_setting
        )
        environment_results = environment_generator.generate_environments()
        generation_data["environment_generation"] = environment_results
        update_pipeline_json()
    else:
        print(f"Environment generation for story {story_id} already completed, skipping...")

    ## Scene Ref Image Prompt Generation
    if not generation_data.get("scene_generation"):
        print(f"Starting reference image prompt generation for story {story_id}...")
        scene_prompt_generator = Script2Scene(
            script_data=script_data,
            story_dir=story_dir,
            deepseek_api_key=deepseek_key,
            gemini_api_keys=gemini_keys,
            gemini_proxy=gemini_proxy_setting,
            openai_key=openai_key
        )
        scene_prompt_results = scene_prompt_generator.generate_scene_prompts()
        generation_data["scene_generation"] = scene_prompt_results
        update_pipeline_json()
    else:
        print(f"Ref image prompt generation for story {story_id} already completed, skipping...")

    ## Video Generation
    print(f"Starting video generation for story {story_id}...")
    video_generator = VideoGenerator(
        script_data=script_data,
        story_dir=story_dir,
        generation_data=generation_data,
        deepseek_key=deepseek_key,
        gemini_keys=gemini_keys,
        gemini_proxy_setting=gemini_proxy_setting
    )
    video_generator.generate_videos()
    video_generator.process_chosen_path()
    update_pipeline_json()
    print(f"Video generation for story {story_id} already completed, skipping...")

    ## Audio Generation
    if not generation_data.get("audio_generation"):
        print("Starting audio generation...")

        chosen_dir = os.path.join(story_dir, 'videos/chosen')
        output_dir = os.path.join(chosen_dir, 'final')
        voices_dir = os.path.join(story_dir, 'voices')
        os.makedirs(output_dir, exist_ok=True)
        voice_generator = CosyVoiceGenerator(
            output_dir=voices_dir,
            script_path=os.path.join(story_dir, 'script.json'),
            pipeline_json_path=os.path.join(story_dir, 'pipeline.json'),                                                                             
        )
        voice_generator.generate_complete_voiceover()
        print("Audio generation completed!")
    else:
        print("Audio generation already completed, skipping...")

    ## Caption and Final Video Generation
    final_movie_path = os.path.join(story_dir, 'videos/chosen/final', f"complete_movie_{story_id}.mp4")
    if not os.path.exists(final_movie_path):
        print("Starting final video generation (captioning and concatenation)...")
        try:
            final_video_gen = FinalVideoGenerator(
                story_dir=story_dir,
                pipeline_json_path=pipeline_json_path
            )
            final_video_gen.generate()
            print(f"Final video generation process finished for story {story_id}.")
        except Exception as e:
            print(f"An error occurred during final video generation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Final movie {final_movie_path} already exists. Skipping final video generation.")

    print(f"Pipeline execution completed for story {story_id}!")

print("All stories processed successfully!")
