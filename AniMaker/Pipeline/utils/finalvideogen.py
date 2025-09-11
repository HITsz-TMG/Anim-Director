import os
import sys
import re
import gc
import json
import torch
import shutil
import subprocess
from moviepy.editor import AudioFileClip, concatenate_audioclips, VideoFileClip


from Tools.caption import VideoCaption

class FinalVideoGenerator:
    def __init__(self, story_dir, pipeline_json_path):
        self.story_dir = story_dir
        self.pipeline_json_path = pipeline_json_path
        self.chosen_dir = os.path.join(self.story_dir, 'videos/chosen')
        self.output_dir = os.path.join(self.chosen_dir, 'final')
        self.story_id = os.path.basename(story_dir) # Extract story ID from path

    def generate(self):
        """Generates captioned clips and concatenates them into a final video."""
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            print(f"Final video directory {self.output_dir} already exists and is not empty. Skipping final video generation.")
            # Check specifically for the final movie file
            final_movie_path_check = os.path.join(self.output_dir, f"complete_movie_{self.story_id}.mp4")
            if os.path.exists(final_movie_path_check):
                print(f"Final movie {final_movie_path_check} already exists.")
                return # Exit if final movie exists
            else:
                print("Final movie not found, proceeding with generation...")
        else:
            print("Starting caption and final video generation...")
            os.makedirs(self.output_dir, exist_ok=True)

        # Step 1: Load pipeline JSON and extract audio segments
        if not os.path.exists(self.pipeline_json_path):
            print(f"Error: Pipeline JSON not found at {self.pipeline_json_path}")
            return

        try:
            with open(self.pipeline_json_path, 'r') as f:
                pipeline_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.pipeline_json_path}")
            return

        # Step 2: Group audio segments by scene and clip
        audio_segments = pipeline_data.get("audio_generation", {}).get("audio_segments", [])
        clip_captions = {}  # {(scene_id, clip_id): "caption text"}
        clip_audio_paths = {}  # {(scene_id, clip_id): [list of audio paths]}

        for segment in audio_segments:
            scene_id = segment.get("scene", 0)
            clip_id = segment.get("clip", 0)
            text = segment.get("text", "")
            audio_path = segment.get("path", "")

            key = (scene_id, clip_id)
            if key not in clip_captions:
                clip_captions[key] = text
            else:
                clip_captions[key] += " " + text

            if key not in clip_audio_paths:
                clip_audio_paths[key] = []

            if audio_path and os.path.exists(audio_path):
                clip_audio_paths[key].append(audio_path)

        print(f"Extracted captions for {len(clip_captions)} clips")

        # Step 3: Process videos in the chosen directory
        if not os.path.exists(self.chosen_dir):
            print(f"Error: Chosen video directory not found at {self.chosen_dir}")
            return

        captioner = VideoCaption()
        processed_videos = []
        # Ensure videos are processed in order (scene_id, then clip_id)
        try:
            chosen_video_files = sorted(
                [f for f in os.listdir(self.chosen_dir) if f.endswith('.mp4') and f.startswith('scene_') and os.path.isfile(os.path.join(self.chosen_dir, f))],
                key=lambda f: (int(re.match(r'scene_(\d+)_clip_(\d+)\.mp4', f).group(1)),
                               int(re.match(r'scene_(\d+)_clip_(\d+)\.mp4', f).group(2)))
            )
        except AttributeError:
            print("Error sorting video files. Ensure filenames match 'scene_X_clip_Y.mp4' pattern.")
            return

        for video_file in chosen_video_files:
            match = re.match(r'scene_(\d+)_clip_(\d+)\.mp4', video_file)
            if not match:
                print(f"Warning: File {video_file} doesn't match expected naming pattern. Skipping.")
                continue

            scene_id = int(match.group(1))
            clip_id = int(match.group(2))
            key = (scene_id, clip_id)

            video_path = os.path.join(self.chosen_dir, video_file)
            captioned_video_path = os.path.join(self.output_dir, f"captioned_{video_file}")
            final_video_path = os.path.join(self.output_dir, f"final_{video_file}")

            # --- Add Caption ---
            caption_text = clip_captions.get(key, "")
            if not caption_text:
                print(f"Warning: No caption found for scene {scene_id}, clip {clip_id}. Using original video.")
                # If no caption, copy original video to avoid issues later
                if not os.path.exists(captioned_video_path):
                    shutil.copy2(video_path, captioned_video_path)
                else:
                    print(f"Using existing captioned file: {captioned_video_path}")

            elif not os.path.exists(captioned_video_path): # Only generate if it doesn't exist
                print(f"Adding caption to scene {scene_id}, clip {clip_id}: {caption_text}")
                try:
                    captioner.add_caption(
                        video_path=video_path,
                        caption_text=caption_text,
                        output_path=captioned_video_path
                    )
                except Exception as e:
                    print(f"Error adding caption to {video_file}: {e}. Using original video.")
                    if not os.path.exists(captioned_video_path): # Ensure file exists even on error
                        shutil.copy2(video_path, captioned_video_path)
            else:
                print(f"Using existing captioned file: {captioned_video_path}")


            # --- Add Audio ---
            audio_paths = clip_audio_paths.get(key, [])
            if not audio_paths:
                print(f"No audio found for scene {scene_id}, clip {clip_id}. Using captioned video.")
                # If no audio, copy captioned video to final path if it doesn't exist
                if not os.path.exists(final_video_path):
                    shutil.copy2(captioned_video_path, final_video_path)
                else:
                    print(f"Using existing final file: {final_video_path}")

            elif not os.path.exists(final_video_path): # Only generate if it doesn't exist
                audio_path_to_add = None
                mixed_audio_path = os.path.join(self.output_dir, f"mixed_audio_{scene_id}_{clip_id}.wav")

                if len(audio_paths) > 1:
                    if not os.path.exists(mixed_audio_path):
                        print(f"Mixing {len(audio_paths)} audio files for scene {scene_id}, clip {clip_id}...")
                        try:
                            audio_clips = [AudioFileClip(path) for path in audio_paths]
                            mixed_clip = concatenate_audioclips(audio_clips)
                            mixed_clip.write_audiofile(mixed_audio_path, codec='pcm_s16le') # Use standard wav codec
                            mixed_clip.close() # Close clips
                            for clip in audio_clips: clip.close()
                            audio_path_to_add = mixed_audio_path
                        except Exception as e:
                            print(f"Error mixing audio for scene {scene_id}, clip {clip_id}: {e}. Falling back to first audio.")
                            audio_path_to_add = audio_paths[0]
                    else:
                        print(f"Using existing mixed audio file: {mixed_audio_path}")
                        audio_path_to_add = mixed_audio_path
                else:
                    audio_path_to_add = audio_paths[0]

                # Add the selected audio to the captioned video
                print(f"Adding audio {os.path.basename(audio_path_to_add)} to {os.path.basename(captioned_video_path)}...")
                try:
                    # Get video duration using ffprobe
                    video_duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                          '-of', 'default=noprint_wrappers=1:nokey=1', captioned_video_path]
                    video_duration = float(subprocess.check_output(video_duration_cmd).decode('utf-8').strip())

                    # Get audio duration using ffprobe
                    audio_duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                          '-of', 'default=noprint_wrappers=1:nokey=1', audio_path_to_add]
                    audio_duration = float(subprocess.check_output(audio_duration_cmd).decode('utf-8').strip())

                    ffmpeg_cmd = ['ffmpeg', '-y'] # Overwrite output

                    # Input video first
                    ffmpeg_cmd.extend(['-i', captioned_video_path])

                    # Input audio second
                    ffmpeg_cmd.extend(['-i', audio_path_to_add])

                    # Mapping and codec settings
                    ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '1:a:0']) # Map video from input 0, audio from input 1
                    ffmpeg_cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']) # Video codec
                    ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '192k']) # Audio codec

                    if audio_duration > video_duration:
                        # Audio is longer: Freeze last frame
                        pad_duration = audio_duration - video_duration
                        print(f"Audio is longer by {pad_duration:.2f}s. Freezing last frame.")
                        # Use filter complex for tpad
                        ffmpeg_cmd.extend([
                            '-filter_complex', f'[0:v]tpad=stop_mode=clone:stop_duration={pad_duration}[v]',
                            '-map', '[v]' # Map the filtered video instead of original
                        ])
                        # Remove the original video map if filter_complex is used for video
                        ffmpeg_cmd.remove('-map')
                        ffmpeg_cmd.remove('0:v:0')
                        # Duration is determined by the audio implicitly
                    else:
                        # Video is longer or equal: Use -shortest
                        print("Video is longer or equal to audio. Using -shortest.")
                        ffmpeg_cmd.append('-shortest')

                    ffmpeg_cmd.append(final_video_path)

                    print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                    subprocess.run(ffmpeg_cmd, check=True)
                    print(f"Generated final clip with audio: {final_video_path}")

                except subprocess.CalledProcessError as e:
                    print(f"Error running FFmpeg for {video_file}: {e}")
                    print(f"FFmpeg stdout: {e.stdout}")
                    print(f"FFmpeg stderr: {e.stderr}")
                    # Fallback: copy captioned video if FFmpeg fails
                    if not os.path.exists(final_video_path):
                        shutil.copy2(captioned_video_path, final_video_path)
                except Exception as e:
                    print(f"Error adding audio to {video_file}: {e}")
                    # Fallback: copy captioned video if any other error occurs
                    if not os.path.exists(final_video_path):
                        shutil.copy2(captioned_video_path, final_video_path)
            else:
                print(f"Using existing final file: {final_video_path}")


            # Add the path of the final processed video (with caption and audio)
            if os.path.exists(final_video_path):
                processed_videos.append(final_video_path)
            else:
                print(f"Warning: Final video path {final_video_path} not found after processing. Skipping concatenation for this clip.")


        # --- Concatenate Final Clips ---
        final_movie_path = os.path.join(self.output_dir, f"complete_movie_{self.story_id}.mp4")

        if not processed_videos:
            print("No processed videos found to concatenate.")
            return

        print(f"Found {len(processed_videos)} videos to concatenate")
        
        # Create a temporary file for FFmpeg concat demuxer (needed for fallback)
        concat_list_path = os.path.join(self.output_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for video_path in processed_videos:
                # Format required by FFmpeg concat demuxer: file 'path/to/file'
                f.write(f"file '{os.path.abspath(video_path)}'\n")
        
        try:
            #--- Primary Method: Use filter_complex for concatenation ---
            print("Attempting concatenation using filter_complex (more robust)...")
            filter_parts = []
            for i in range(len(processed_videos)):
                filter_parts.append(f"[{i}:v:0][{i}:a:0]")
            
            inputs = []
            for vid in processed_videos:
                inputs.extend(['-i', vid])
            
            filter_complex = ''.join(filter_parts) + f"concat=n={len(processed_videos)}:v=1:a=1[outv][outa]"
            
            final_cmd = [
                'ffmpeg', '-y'
            ] + inputs + [
                '-filter_complex', filter_complex,
                '-map', '[outv]', '-map', '[outa]',
                # Consistent encoding settings
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-vsync', 'cfr', # Ensure constant frame rate
                '-max_muxing_queue_size', '1024', # Add buffer
                final_movie_path
            ]
            
            print(f"Running FFmpeg filter_complex command: {' '.join(final_cmd)}")
            result = subprocess.run(final_cmd, check=False, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Warning: FFmpeg filter_complex returned error code {result.returncode}")
                print(f"Error details: {result.stderr}")
                # Raise an error to trigger the fallback
                raise subprocess.CalledProcessError(result.returncode, final_cmd, output=result.stdout, stderr=result.stderr)
            else:
                print(f"Final complete video created using filter_complex: {final_movie_path}")

        except Exception as e:
            print(f"Filter_complex concatenation failed: {e}")
            print("Attempting fallback concatenation using concat demuxer...")
            try:
                # --- Fallback Method: Use concat demuxer ---
                concat_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',  # Allow absolute paths
                    '-i', concat_list_path,
                    # Re-encode video and audio with consistent parameters
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-max_muxing_queue_size', '1024',
                    final_movie_path
                ]
                
                print(f"Running FFmpeg concat demuxer command: {' '.join(concat_cmd)}")
                result = subprocess.run(concat_cmd, check=False, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"Warning: FFmpeg concat demuxer also failed. Error code {result.returncode}")
                    print(f"Error details: {result.stderr}")
                    # Raise error if fallback also fails
                    raise subprocess.CalledProcessError(result.returncode, concat_cmd, output=result.stdout, stderr=result.stderr)
                else:
                    print(f"Final complete video created using concat demuxer (fallback): {final_movie_path}")

            except Exception as e2:
                print(f"All concatenation approaches failed: {e2}")
                # Handle the failure case appropriately, maybe skip final video or raise exception

        # finally:
        # Clean up temporary concat list
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
        
        # Release memory
        gc.collect()
        torch.cuda.empty_cache()

