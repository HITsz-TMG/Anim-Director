import os
import sys
sys.path.append('Tools/CosyVoice')
sys.path.append('Tools/CosyVoice/third_party/Matcha-TTS')
import re
import json
import torch
import torchaudio
from Tools.gemini_api import GeminiAPI
from Tools.deepseek_api import DeepSeekAPI
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class CosyVoiceGenerator:
    def __init__(self, 
                 model_path='Tools/CosyVoice/pretrained_models/CosyVoice2-0.5B',
                 voice_prompt_dir='Tools/CosyVoice/asset/speech',
                 output_dir='Pipeline/voices',
                 script_path='Pipeline/script.json',
                 pipeline_json_path='Pipeline/pipeline.json',
                 gemini_api_keys=["your_gemini_key_1_here", "your_gemini_key_2_here", "your_gemini_key_3_here"],
                 gemini_proxy='your_proxy_here',
                 deepseek_api_key="your_deepseek_key_here"):
        
        # Initialize output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize API clients
        self.gemini = GeminiAPI(api_keys=gemini_api_keys, proxy=gemini_proxy)
        self.deepseek_api = DeepSeekAPI(api_key=deepseek_api_key)
        
        # Initialize CosyVoice model
        self.cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
        
        # Load voice prompts
        self.voice_prompts = {
            "boy": load_wav(os.path.join(voice_prompt_dir, 'boy.wav'), 16000),
            "girl": load_wav(os.path.join(voice_prompt_dir, 'girl.mp3'), 16000),
            "youngman": load_wav(os.path.join(voice_prompt_dir, 'youngman.mp3'), 16000),
            "youngwoman": load_wav(os.path.join(voice_prompt_dir, 'youngwoman.mp3'), 16000),
            "man": load_wav(os.path.join(voice_prompt_dir, 'man.mp3'), 16000),
            "woman": load_wav(os.path.join(voice_prompt_dir, 'woman.wav'), 16000),
            "elderly": load_wav(os.path.join(voice_prompt_dir, 'elderly.mp3'), 16000),
            "narrative": load_wav(os.path.join(voice_prompt_dir, 'narrative_low.mp3'), 16000)
        }
        
        # Set default voice prompt
        self.default_prompt_speech = self.voice_prompts["narrative"]
        
        # Store script and pipeline paths
        self.script_path = script_path
        self.pipeline_json_path = pipeline_json_path
        
        # Load pipeline.json if it exists
        if os.path.exists(self.pipeline_json_path):
            with open(self.pipeline_json_path, 'r') as f:
                self.pipeline_data = json.load(f)
        else:
            self.pipeline_data = {}
            
        # Initialize audio_generation data
        self.audio_generation_data = {
            "voiceover_script_prompt": "",
            "voiceover_script": "",
            "script_path": self.script_path,
            "output_directory": self.output_dir,
            "audio_segments": []
        }
        
        self.script_data = None
        self.audio_segments = []

    def load_script(self, script_path=None):
        """Load the animation script from file"""
        if script_path:
            self.script_path = script_path
            
        with open(self.script_path, 'r') as f:
            self.script_data = json.load(f)
            
        return self.script_data
    
    def generate_voiceover_script(self, force_generate=False):
        """Generate or load the voiceover script"""
        if not self.script_data:
            self.load_script()
            
        # Concatenate all clip descriptions
        script_lines = []
        for scene in self.script_data['script']:
            for clip in scene['clips']:
                script_lines.append(clip['description'])
        
        script = "\n".join(script_lines)
        script_lines_num = len(script_lines)
        
        prompt = f"""
        Below is a script for an animation. Each line describes a shot lasting approximately 5 seconds. 

        {script}

        Please create a professional voice-over script following these guidelines:
        1. Each line corresponds to one shot from the original script, keeping appropriate length for 5-second shots (10-13 words per line). Thus your script should have exactly {script_lines_num} lines.
        2. Format all lines using the pattern: [emotion:voice_type] content, where:
           - emotion is the emotional tone (e.g., happy, sad, curious)
           - voice_type is one of: narrative, boy, female, younglady
        3. For narration, use voice_type "narrative"
        4. For character dialogue, wrap spoken words in <speak>[emotion:voice_type] </speak> tags and use:
           - "boy" for boy characters
           - "girl" for girl characters
           - "youngman" for young male characters
           - "youngwoman" for young female characters
           - "woman" for adult female characters
           - "man" for adult male characters
           - "elderly" for elderly characters

        5. Preserve the original sequence of narration and dialogue

        Example formats:
        - Narration: [thoughtful:narrative] The sun cast long shadows across the empty field.
        - Boy dialogue: <speak>[curious:boy] What is this tiny plant growing here?</speak>
        - Mixed: [gentle:narrative] She knelt beside him and said, <speak>[excited:female] Look how much you've grown!</speak>

        Use appropriate emotions that match each moment in the story.
        """
        
        # Save the prompt to the audio_generation data
        self.audio_generation_data["voiceover_script_prompt"] = prompt
        
        # Check if voiceover_script.txt already exists
        voiceover_script_path = os.path.join(self.output_dir, 'voiceover_script.txt')
        
        # Flag to track if we need to generate (either forced or first time or line count mismatch)
        need_to_generate = force_generate or not os.path.exists(voiceover_script_path)
        
        # If file exists and we're not forcing regeneration, check line count
        if os.path.exists(voiceover_script_path) and not force_generate:
            with open(voiceover_script_path, 'r', encoding='utf-8') as f:
                existing_script = f.read()
                
            # Parse to count valid voice lines
            existing_lines = [line for line in existing_script.strip().split('\n') 
                             if line and '[' in line and ']' in line]
                
            # Check if line count matches expected
            if len(existing_lines) != script_lines_num:
                print(f"Warning: Existing voice-over script has {len(existing_lines)} lines but expected {script_lines_num}.")
                print("Regenerating voice-over script to match expected line count.")
                need_to_generate = True
            else:
                print(f"Found existing voice-over script at {voiceover_script_path} with correct line count, using it.")
                answer = existing_script
        
        # Generate new script if needed
        if need_to_generate:
            print("Generating new voice-over script...")
            answer = self.deepseek_api.generate_from_text(prompt)
            
            # Validate line count in generated script
            generated_lines = [line for line in answer.strip().split('\n') 
                              if line and '[' in line and ']' in line]
            
            # If generated script doesn't have correct number of lines, try again
            attempts = 1
            max_attempts = 3
            while len(generated_lines) != script_lines_num and attempts < max_attempts:
                print(f"Generated script has {len(generated_lines)} lines but expected {script_lines_num}.")
                print(f"Attempt {attempts+1}/{max_attempts} to generate script with correct line count...")
                
                # Add more explicit instruction about line count
                refined_prompt = prompt + f"\n\nIMPORTANT: Your response MUST contain exactly {script_lines_num} formatted voice-over lines (one line per shot)."
                
                answer = self.deepseek_api.generate_from_text(refined_prompt)
                generated_lines = [line for line in answer.strip().split('\n') 
                                  if line and '[' in line and ']' in line]
                attempts += 1
            
            # Save the final script
            with open(voiceover_script_path, 'w', encoding='utf-8') as f:
                f.write(answer)
                print(f"Saved complete voice-over script to {voiceover_script_path}")
            
            if len(generated_lines) != script_lines_num:
                print(f"Warning: After {max_attempts} attempts, still couldn't generate exactly {script_lines_num} lines.")
                print(f"Proceeding with {len(generated_lines)} lines.")
        
        # Save the generated script to the audio_generation data
        self.audio_generation_data["voiceover_script"] = answer
        
        # Parse the answer to extract voice lines with emotions
        voice_lines = []
        for line in answer.strip().split('\n'):
            if line and '[' in line and ']' in line:
                voice_lines.append(line)
                
        self.voice_lines = voice_lines
        
        # Update pipeline.json with voiceover script data
        self.update_pipeline_json()
        
        return voice_lines

    def extract_emotion_voice_and_clean(self, text):
        """Extract emotion and voice type from text in format [emotion:voice_type]"""
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, text)
        
        if match:
            tag_content = match.group(1)
            # Default values
            emotion = "neutral"
            voice_type = "narrative"
            
            # Check if there's a voice type specified
            if ":" in tag_content:
                parts = tag_content.split(":")
                emotion = parts[0].strip()
                voice_type = parts[1].strip()
            else:
                # Only emotion is specified
                emotion = tag_content.strip()
            
            # Remove the tag from the text
            cleaned_text = re.sub(pattern, '', text, 1).strip()
            return emotion, voice_type, cleaned_text
        
        return "neutral", "narrative", text.strip()  # Default values and original text

    def verify_audio_matches_text(self, audio_path, expected_text):
        """Verify if generated audio matches the expected text using Gemini API"""
        prompt = f"Does the audio say the following text: '{expected_text}'? Answer only with 'Yes' or 'No'."
        audio_result = self.gemini.generate_from_videos([audio_path], prompt)
        
        # Check if the result contains "Yes"
        return "yes" in audio_result.lower()

    def generate_verified_audio(self, text, emotion, voice_prompt, output_path, voice_type="narrative"):
        """Generate audio with verification and retry logic"""
        # Parse base filename for failed versions
        base_path, ext = os.path.splitext(output_path)
        
        # Track generation attempts for this segment
        audio_attempt = {
            "text": text,
            "emotion": emotion,
            "output_path": output_path,
            "attempts": []
        }
        
        # Skip emotion formatting for narrative voice type
        formatted_emotion = "" if voice_type == "narrative" else f"#{emotion}#"
        
        # First attempt with specified emotion (or no emotion for narrative)
        for j, result in enumerate(self.cosyvoice.inference_instruct2(text, formatted_emotion, voice_prompt, stream=False)):
            torchaudio.save(output_path, result['tts_speech'], self.cosyvoice.sample_rate)
            
            # Record the first attempt
            attempt_result = self.verify_audio_matches_text(output_path, text)
            audio_attempt["attempts"].append({
                "attempt": 1,
                "emotion_format": formatted_emotion,
                "success": attempt_result
            })
            
            # Verify audio
            if attempt_result:
                print(f"✓ Audio verification passed for: '{text}'")
                return True, audio_attempt
            
            # First retry with the same parameters - save with failed1 suffix
            print(f"✗ Audio verification failed, retrying with same parameters for: '{text}'")
            failed1_path = f"{base_path}_failed1{ext}"
            for j, result in enumerate(self.cosyvoice.inference_instruct2(text, formatted_emotion, voice_prompt, stream=False)):
                torchaudio.save(failed1_path, result['tts_speech'], self.cosyvoice.sample_rate)
                
                # Record the second attempt
                retry_result = self.verify_audio_matches_text(failed1_path, text)
                audio_attempt["attempts"].append({
                    "attempt": 2,
                    "emotion_format": formatted_emotion,
                    "success": retry_result
                })
                
                if retry_result:
                    print(f"✓ Audio verification passed on retry for: '{text}'")
                    # Copy successful attempt to original path
                    torchaudio.save(output_path, result['tts_speech'], self.cosyvoice.sample_rate)
                    return True, audio_attempt
                
                # Second retry with empty emotion - save with failed2 suffix
                print(f"✗ Audio verification failed again, retrying with empty emotion for: '{text}'")
                empty_emotion = ""
                failed2_path = f"{base_path}_failed2{ext}"
                for j, result in enumerate(self.cosyvoice.inference_instruct2(text, empty_emotion, voice_prompt, stream=False)):
                    torchaudio.save(failed2_path, result['tts_speech'], self.cosyvoice.sample_rate)
                    
                    # Record the third attempt
                    final_result = self.verify_audio_matches_text(failed2_path, text)
                    audio_attempt["attempts"].append({
                        "attempt": 3,
                        "emotion_format": empty_emotion,
                        "success": final_result
                    })
                    
                    if final_result:
                        print(f"✓ Audio verification passed with empty emotion for: '{text}'")
                        # Copy successful attempt to original path
                        torchaudio.save(output_path, result['tts_speech'], self.cosyvoice.sample_rate)
                        return True, audio_attempt
                    else:
                        print(f"✗ Audio verification failed with all attempts for: '{text}'")
                        # Keep all failed attempts on disk for inspection
                        return False, audio_attempt
        
        return False, audio_attempt
    
    def validate_and_fix_script_length(self):
        """Check voiceover script lines and regenerate those under 10 or exceeding 13 words"""
        if not hasattr(self, 'voice_lines'):
            print("No voice lines found. Generate voiceover script first.")
            return []
            
        # Set maximum iteration count
        max_iterations = 3
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Lines that need to be regenerated (under 10 or exceed 13 words)
            lines_to_fix = []
            
            for i, line in enumerate(self.voice_lines):
                # Extract the actual text content
                emotion, voice_type, clean_text = self.extract_emotion_voice_and_clean(line)
                
                # Remove <speak> tags and any content within square brackets for word counting
                text_for_counting = re.sub(r'</?speak>', '', clean_text)
                text_for_counting = re.sub(r'\[.*?\]', '', text_for_counting)
                
                # Count words (split by whitespace)
                word_count = len(text_for_counting.split())
                
                # Check if word count is outside the desired range (10-13 words)
                if word_count < 10 or word_count > 13:
                    reason = "too short" if word_count < 10 else "too long"
                    print(f"Line {i+1} has {word_count} words ({reason}): '{clean_text}'")
                    lines_to_fix.append((i, line, clean_text, word_count, reason))
            
            # Break the loop if no lines need fixing
            if not lines_to_fix:
                print(f"All lines have appropriate length after {iterations} iterations.")
                break
                
            print(f"Iteration {iterations}/{max_iterations}: Found {len(lines_to_fix)} lines that need to be adjusted. Regenerating...")
            
            # If there are lines to fix, regenerate them
            if lines_to_fix:
                for idx, original_line, content, word_count, reason in lines_to_fix:
                    # Extract emotion and voice type from the original line
                    emotion, voice_type, _ = self.extract_emotion_voice_and_clean(original_line)
                    
                    adjustment = "longer" if reason == "too short" else "shorter"
                    
                    # Create a prompt to adjust the line
                    prompt = f"""
                    Please rewrite the following voice line to be {adjustment} (to 10-13 words) while 
                    maintaining the same meaning and style. Keep the emotional tone.
                    
                    Original line ({word_count} words): {content}
                    
                    Format the output:
                    1. Use the pattern: [emotion:voice_type] content, where:
                    - emotion is the emotional tone (e.g., happy, sad, curious)
                    - voice_type is one of: narrative, boy, female, younglady
                    2. For narration, use voice_type "narrative"
                    3. For character dialogue, wrap speech in <speak>[emotion:voice_type] </speak> tags and use:
                    - "boy" for boy characters
                    - "female" for adult female characters
                    - "younglady" for young female characters
                    4. Preserve the original sequence of narration and dialogue

                    Example formats:
                    - Narration: [thoughtful:narrative] The sun cast long shadows across the empty field.
                    - Boy dialogue: <speak>[curious:boy] What is this tiny plant growing here?</speak>
                    - Mixed: [gentle:narrative] She knelt beside him and said, <speak>[excited:female] Look how much you've grown!</speak>

                    Note that words between <speak> tags and square brackets will not be counted in the word count.
                    Directly give the adjusted line without any guide words or markdown.
                    """
                    
                    # Generate an adjusted version using the API
                    adjusted_response = self.deepseek_api.generate_from_text(prompt)
                    
                    # Extract the adjusted line from the response
                    adjusted_line = adjusted_response.strip()
                    
                    # Ensure it has the correct format with emotion/voice tags
                    if '[' in adjusted_line and ']' in adjusted_line:
                        # Replace the original line with the adjusted version
                        self.voice_lines[idx] = adjusted_line
                        
                        # Check the word count of the new line
                        _, _, new_clean_text = self.extract_emotion_voice_and_clean(adjusted_line)
                        new_text_for_counting = re.sub(r'</?speak>', '', new_clean_text)
                        new_text_for_counting = re.sub(r'\[.*?\]', '', new_text_for_counting)
                        new_word_count = len(new_text_for_counting.split())
                        
                        print(f"Adjusted line {idx+1} from {word_count} to {new_word_count} words")
                        print(f"  Old: {content}")
                        print(f"  New: {new_clean_text}")
                    else:
                        print(f"Failed to properly format adjusted line {idx+1}. Keeping original.")
                
                # Update the voiceover script file with the modified lines
                voiceover_script_path = os.path.join(self.output_dir, 'voiceover_script.txt')
                with open(voiceover_script_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.voice_lines))
                    print(f"Updated voiceover script with adjusted lines at {voiceover_script_path}")
        
        if iterations == max_iterations and lines_to_fix:
            print(f"Warning: Reached maximum {max_iterations} iterations with {len(lines_to_fix)} lines still needing adjustment.")
            print("Proceeding with current version of script.")
            
        return self.voice_lines

    def check_and_adjust_audio_duration(self):
        """Check duration of generated audio files and adjust if too long or short"""
        if not hasattr(self, 'audio_segments') or not self.audio_segments:
            print("No audio segments found. Generate voice segments first.")
            return []
        
        # Create a list to store segments that need adjustment
        segments_to_adjust = []
        
        # Define duration thresholds
        min_duration = 3.5  # seconds
        max_duration = 5.5  # seconds
        
        print("Checking audio durations against thresholds...")
        print(f"Minimum duration: {min_duration} seconds")
        print(f"Maximum duration: {max_duration} seconds")
        
        # Check each audio segment
        for segment in self.audio_segments:
            audio_path = segment['path']
            
            # Skip if file doesn't exist
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                continue
                
            # Get audio duration
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            # Get original text and info for this segment
            original_segment_data = next((s for s in self.audio_generation_data["audio_segments"] 
                                        if s["path"] == audio_path), None)
            
            if original_segment_data:
                text = original_segment_data["text"]
                voice_type = original_segment_data["voice_type"]
                emotion = original_segment_data["emotion"]
                
                # Check if duration is outside acceptable range
                if duration < min_duration or duration > max_duration:
                    reason = "too short" if duration < min_duration else "too long"
                    word_count = len(text.split())
                    target_count = int(word_count * 1.2) if duration < min_duration else int(word_count * 0.8)
                    
                    print(f"Audio segment '{audio_path}' is {reason} ({duration:.2f} seconds)")
                    print(f"  Text: '{text}' ({word_count} words)")
                    print(f"  Adjusting to target {target_count} words")
                    
                    segments_to_adjust.append({
                        'path': audio_path,
                        'text': text,
                        'duration': duration,
                        'reason': reason,
                        'original_word_count': word_count,
                        'target_word_count': target_count,
                        'voice_type': voice_type,
                        'emotion': emotion,
                        'segment_data': original_segment_data
                    })
        
        # Process segments that need adjustment
        if segments_to_adjust:
            print(f"Found {len(segments_to_adjust)} audio segments that need duration adjustment")
            
            for segment_info in segments_to_adjust:
                adjustment = "longer" if segment_info['reason'] == "too short" else "shorter"
                target_word_count = segment_info['target_word_count']
                
                # Prepare prompt for text adjustment
                prompt = f"""
                Please rewrite the following voice line to be {adjustment} (to approximately {target_word_count} words) while 
                maintaining the same meaning and style. Keep the emotional tone.
                
                Original line ({segment_info['original_word_count']} words): {segment_info['text']}
                
                Directly give the adjusted text without any formatting, tags, or additional comments.
                """
                
                # Generate adjusted text
                adjusted_response = self.deepseek_api.generate_from_text(prompt)
                adjusted_text = adjusted_response.strip()
                
                print(f"Adjusted text from {segment_info['original_word_count']} to approx. {target_word_count} words:")
                print(f"  Old: {segment_info['text']}")
                print(f"  New: {adjusted_text}")
                
                # Select the appropriate voice prompt
                prompt_speech = self.voice_prompts.get(segment_info['voice_type'], self.default_prompt_speech)
                
                # Generate new audio with the adjusted text
                success, audio_attempt = self.generate_verified_audio(
                    adjusted_text, 
                    segment_info['emotion'], 
                    prompt_speech, 
                    segment_info['path'], 
                    segment_info['voice_type']
                )
                
                # Update segment data with the new text and attempts
                if success:
                    # Check new duration
                    info = torchaudio.info(segment_info['path'])
                    new_duration = info.num_frames / info.sample_rate
                    
                    print(f"✓ Generated new audio with duration {new_duration:.2f} seconds (was {segment_info['duration']:.2f})")
                    
                    # Update the segment data in audio_generation_data
                    segment_info['segment_data']['text'] = adjusted_text
                    segment_info['segment_data']['generation_attempts'].extend(audio_attempt['attempts'])
                    segment_info['segment_data']['duration_adjusted'] = True
                    segment_info['segment_data']['old_duration'] = segment_info['duration']
                    segment_info['segment_data']['new_duration'] = new_duration
                else:
                    print(f"✗ Failed to generate new audio for adjusted text")
        
        # Update pipeline.json if any adjustments were made
        if segments_to_adjust:
            self.update_pipeline_json()
            print(f"Updated audio data in pipeline.json with duration adjustments")
            
        return segments_to_adjust

    def generate_voice_segments(self, force_generate=False):
        """Generate individual voice segments for each line in the script"""
        if not hasattr(self, 'voice_lines'):
            self.generate_voiceover_script(force_generate)
            # Add validation step to check and fix lines exceeding word limit
            self.validate_and_fix_script_length()
            
        if not self.script_data:
            self.load_script()
            
        # Reset audio segments
        self.audio_segments = []
        self.audio_generation_data["audio_segments"] = []
        
        # Create mapping from flat line index to scene/clip indices
        line_to_scene_clip = {}
        flat_line_idx = 0
        
        for scene_idx, scene in enumerate(self.script_data['script']):
            for clip_idx, clip in enumerate(scene['clips']):
                line_to_scene_clip[flat_line_idx] = {
                    'scene': scene_idx + 1,  # 1-indexed
                    'clip': clip_idx + 1     # 1-indexed
                }
                flat_line_idx += 1
        
        # Process each line from the script
        for line_idx, line in enumerate(self.voice_lines):
            segment_counter = 0
            
            # Get scene and clip numbers for this line
            scene_clip_info = line_to_scene_clip.get(line_idx, {'scene': 0, 'clip': 0})
            scene_num = scene_clip_info['scene']
            clip_num = scene_clip_info['clip']
            
            # Check if line contains dialogue in <speak> tags
            if '<speak>' in line and '</speak>' in line:
                # We need to maintain the original order of text and dialogue
                # First, identify all speak tags and their positions
                speak_positions = []
                for match in re.finditer(r'<speak>|</speak>', line):
                    speak_positions.append((match.start(), match.group()))
                
                # Process the line in segments
                last_pos = 0
                segments = []
                
                # Group by pairs of <speak> and </speak>
                i = 0
                while i < len(speak_positions):
                    if speak_positions[i][1] == '<speak>':
                        # Text before the speak tag
                        if last_pos < speak_positions[i][0]:
                            pre_text = line[last_pos:speak_positions[i][0]]
                            if pre_text.strip():
                                segments.append(('narration', pre_text))
                        
                        # Find the closing tag
                        if i + 1 < len(speak_positions) and speak_positions[i+1][1] == '</speak>':
                            # Extract the dialogue
                            dialogue_start = speak_positions[i][0] + 7  # Length of <speak>
                            dialogue_end = speak_positions[i+1][0]
                            dialogue = line[dialogue_start:dialogue_end]
                            if dialogue.strip():
                                segments.append(('dialogue', dialogue))
                            
                            last_pos = speak_positions[i+1][0] + 8  # Length of </speak>
                            i += 2  # Skip the closing tag
                        else:
                            # Malformed tags, skip this one
                            last_pos = speak_positions[i][0] + 7
                            i += 1
                    else:
                        # Unexpected closing tag, skip it
                        last_pos = speak_positions[i][0] + 8
                        i += 1
                
                # Text after the last speak tag
                if last_pos < len(line):
                    post_text = line[last_pos:]
                    if post_text.strip():
                        segments.append(('narration', post_text))
                
                # Now process each segment in order
                for segment_type, segment_text in segments:       
                    # Extract emotion, voice type, and clean text
                    emotion, voice_type, clean_text = self.extract_emotion_voice_and_clean(segment_text)
                    
                    if clean_text:  # Only process if there's actual content
                        # Select the appropriate voice prompt
                        prompt_speech = self.voice_prompts.get(voice_type, self.default_prompt_speech)
                        
                        # Generate voice for this segment with verification using new naming pattern
                        output_path = os.path.join(
                            self.output_dir, 
                            f'scene_{scene_num}_clip_{clip_num}_{segment_type}_{segment_counter}.wav'
                        )
                        success, audio_attempt = self.generate_verified_audio(clean_text, emotion, prompt_speech, output_path, voice_type)
                        
                        # Create segment data to store in pipeline.json
                        segment_data = {
                            'line': line_idx,
                            'scene': scene_num,
                            'clip': clip_num,
                            'segment': segment_counter,
                            'type': segment_type,
                            'text': clean_text,
                            'emotion': emotion,
                            'voice_type': voice_type,
                            'path': output_path,
                            'success': success,
                            'generation_attempts': audio_attempt["attempts"]
                        }
                        
                        # Add to audio_generation_data
                        self.audio_generation_data["audio_segments"].append(segment_data)
                        
                        if success:
                            print(f"Saved {segment_type} for scene {scene_num}, clip {clip_num}: '{clean_text}' with emotion '{emotion}' and voice '{voice_type}'")
                            
                            # Store information for ordering
                            self.audio_segments.append({
                                'line': line_idx,
                                'scene': scene_num,
                                'clip': clip_num,
                                'segment': segment_counter,
                                'type': segment_type,
                                'path': output_path
                            })
                            
                            segment_counter += 1
                        else:
                            print(f"Failed to generate satisfactory audio for: '{clean_text}'")
            else:
                # No dialogue tags, process as single narration
                emotion, voice_type, clean_text = self.extract_emotion_voice_and_clean(line)
                
                # Select the appropriate voice prompt
                prompt_speech = self.voice_prompts.get(voice_type, self.default_prompt_speech)
                
                # Generate voice for this line with verification
                output_path = os.path.join(
                    self.output_dir, 
                    f'scene_{scene_num}_clip_{clip_num}_narration_{segment_counter}.wav'
                )
                success, audio_attempt = self.generate_verified_audio(clean_text, emotion, prompt_speech, output_path, voice_type)
                
                # Create segment data to store in pipeline.json
                segment_data = {
                    'line': line_idx,
                    'scene': scene_num,
                    'clip': clip_num,
                    'segment': segment_counter,
                    'type': 'narration',
                    'text': clean_text,
                    'emotion': emotion,
                    'voice_type': voice_type,
                    'path': output_path,
                    'success': success,
                    'generation_attempts': audio_attempt["attempts"]
                }
                
                # Add to audio_generation_data
                self.audio_generation_data["audio_segments"].append(segment_data)
                
                if success:
                    print(f"Saved narration for scene {scene_num}, clip {clip_num}: '{clean_text}' with emotion '{emotion}' and voice '{voice_type}'")
                    
                    # Store information for ordering
                    self.audio_segments.append({
                        'line': line_idx,
                        'scene': scene_num,
                        'clip': clip_num,
                        'segment': segment_counter,
                        'type': 'narration',
                        'path': output_path
                    })
                    
                    segment_counter += 1
                else:
                    print(f"Failed to generate satisfactory audio for: '{clean_text}'")
        
        # Update pipeline.json with generated segments
        self.update_pipeline_json()
        
        print("Voice generation complete.")
        return self.audio_segments

    def concatenate_audio_files(self, file_list):
        """Load and concatenate audio files"""
        waveforms = []
        for filepath in file_list:
            waveform, sr = torchaudio.load(filepath)
            assert sr == self.cosyvoice.sample_rate, f"Sample rate mismatch: {sr} vs {self.cosyvoice.sample_rate}"
            waveforms.append(waveform)
        
        # Concatenate along time dimension (dim=1)
        if waveforms:
            return torch.cat(waveforms, dim=1)
        else:
            return None

    def concatenate_segments(self):
        """Concatenate segments into complete audio files"""
        if not self.audio_segments:
            print("No audio segments found. Run generate_voice_segments() first.")
            return
            
        print("Concatenating all voice clips into a complete audio file...")
        
        # Sort segments by line, then by segment number within each line
        self.audio_segments.sort(key=lambda x: (x['line'], x['segment']))
        ordered_files = [segment['path'] for segment in self.audio_segments]
        
        # Track concatenated files
        concatenated_files = []
        
        # Concatenate all files and save the result
        if ordered_files:
            combined_waveform = self.concatenate_audio_files(ordered_files)
            if combined_waveform is not None:
                combined_output_path = os.path.join(self.output_dir, 'complete_voiceover.wav')
                torchaudio.save(combined_output_path, combined_waveform, self.cosyvoice.sample_rate)
                print(f"Successfully saved complete voiceover to {combined_output_path}")
                concatenated_files.append({
                    "type": "complete",
                    "path": combined_output_path,
                    "source_files": ordered_files
                })
            else:
                print("No valid audio files found to concatenate")
        else:
            print("No voice files found to concatenate")
        
        # Create separate scene-based audio files
        scene_clips = {}
        for segment in self.audio_segments:
            scene_num = segment['scene']
            if scene_num not in scene_clips:
                scene_clips[scene_num] = []
            scene_clips[scene_num].append(segment['path'])
        
        # Concatenate and save audio for each scene
        for scene_num, files in scene_clips.items():
            scene_waveform = self.concatenate_audio_files(files)
            if scene_waveform is not None:
                scene_output_path = os.path.join(self.output_dir, f'scene_{scene_num}_voiceover.wav')
                torchaudio.save(scene_output_path, scene_waveform, self.cosyvoice.sample_rate)
                print(f"Saved voiceover for Scene {scene_num} to {scene_output_path}")
                concatenated_files.append({
                    "type": "scene",
                    "scene": scene_num,
                    "path": scene_output_path,
                    "source_files": files
                })
        
        # Add concatenated files to audio_generation_data
        self.audio_generation_data["concatenated_files"] = concatenated_files
        
        # Update pipeline.json with concatenated files
        self.update_pipeline_json()
        
        print("Audio concatenation complete.")
        return self.output_dir

    def update_pipeline_json(self):
        """Update the pipeline.json file with audio generation data"""
        # Add audio_generation data to pipeline data
        self.pipeline_data["audio_generation"] = self.audio_generation_data
        
        # Write updated data to pipeline.json
        with open(self.pipeline_json_path, 'w') as f:
            json.dump(self.pipeline_data, f, indent=4)
        
        print(f"Updated audio generation data in {self.pipeline_json_path}")

    def generate_complete_voiceover(self, force_generate=False):
        """Main method to generate the complete voiceover"""
        self.load_script()
        self.generate_voiceover_script(force_generate)
        # Add validation step to check and fix lines exceeding word limit
        self.validate_and_fix_script_length()
        self.generate_voice_segments(force_generate)
        # Check and adjust audio durations
        self.check_and_adjust_audio_duration()
        result = self.concatenate_segments()
        
        # Make sure to update pipeline.json with final data
        self.update_pipeline_json()
        
        return result


# # Example usage:
# if __name__ == "__main__":
#     voice_generator = CosyVoiceGenerator()
#     voice_generator.generate_complete_voiceover()
