import os
import sys

sys.path.append('Tools/CosyVoice/third_party/Matcha-TTS')
import re
import json
import torch
import torchaudio
from Tools.gemini_api import GeminiAPI
from Tools.deepseek_api import DeepSeekAPI
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Initialize GeminiAPI for verification
gemini = GeminiAPI(
    api_key="your_gemini_key_here",
    proxy='your_proxy_here'
)
deepseek_api = DeepSeekAPI(api_key="your_deepseek_key_here")

# Load script.json and extract descriptions
script_path = 'Pipeline/script.json'
with open(script_path, 'r') as f:
    script_data = json.load(f)

# Concatenate all clip descriptions
script_lines = []
for scene in script_data['script']:
    for clip in scene['clips']:
        script_lines.append(clip['description'])

script = "\n".join(script_lines)

prompt = f"""
Below is a script for an animation. Each line describes a shot lasting approximately 5 seconds.

{script}

Please create a professional voice-over script following these guidelines:
1. Each line corresponds to one shot from the original script, keeping appropriate length for 5-second shots (approximately 10-15 words per line)
2. Format all lines using the pattern: [emotion:voice_type] content, where:
   - emotion is the emotional tone (e.g., happy, sad, curious)
   - voice_type is one of: narrative, boy, female, younglady
3. For narration, use voice_type "narrative"
4. For character dialogue, wrap speech in <speak> </speak> tags and use:
   - "boy" for boy characters
   - "female" for adult female characters
   - "younglady" for young female characters
5. Preserve the original sequence of narration and dialogue

Example formats:
- Narration: [thoughtful:narrative] The sun cast long shadows across the empty field.
- Boy dialogue: [curious:boy] <speak>What is this tiny plant growing here?</speak>
- Mixed: [gentle:narrative] She knelt beside him and said, <speak>[excited:female] Look how much you've grown!</speak>

Use appropriate emotions that match each moment in the story.
"""

# Ensure voices directory exists
voices_dir = 'Pipeline/voices'
os.makedirs(voices_dir, exist_ok=True)

# Check if voiceover_script.txt already exists
voiceover_script_path = os.path.join(voices_dir, 'voiceover_script.txt')
if os.path.exists(voiceover_script_path):
    print(f"Found existing voice-over script at {voiceover_script_path}, using it instead of generating new one")
    with open(voiceover_script_path, 'r', encoding='utf-8') as f:
        answer = f.read()
else:
    print("No existing voice-over script found, generating new one...")
    answer = deepseek_api.generate_from_text(prompt)
    with open(voiceover_script_path, 'w', encoding='utf-8') as f:
        f.write(answer)
        print(f"Saved complete voice-over script to {voiceover_script_path}")

# Parse the answer to extract voice lines with emotions
voice_lines = []
for line in answer.strip().split('\n'):
    if line and '[' in line and ']' in line:
        voice_lines.append(line)

# Initialize CosyVoice
cosyvoice = CosyVoice2('Tools/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# Load all available voice prompts
voice_prompts = {
    "boy": load_wav('Tools/CosyVoice/asset/speech/boy.wav', 16000),
    "female": load_wav('Tools/CosyVoice/asset/speech/female.wav', 16000),
    "younglady": load_wav('Tools/CosyVoice/asset/speech/younglady.wav', 16000),
    "narrative": load_wav('Tools/CosyVoice/asset/speech/narrative.wav', 16000)
}

# Default voice prompt if no specific type is found
default_prompt_speech_16k = voice_prompts["narrative"]

# Function to extract emotion from text in format [emotion] and clean the text
def extract_emotion_and_clean(text):
    emotion_match = re.search(r'\[(.*?)\]', text)
    if emotion_match:
        emotion = emotion_match.group(1)
        # Remove the emotion tag from the text
        cleaned_text = re.sub(r'\[.*?\]', '', text, 1).strip()
        return emotion, cleaned_text
    return "", text.strip()  # Default emotion and original text

# Function to extract emotion and voice type from text in format [emotion:voice_type]
def extract_emotion_voice_and_clean(text):
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

# Function to verify if generated audio matches the expected text
def verify_audio_matches_text(audio_path, expected_text):
    """
    Verify if generated audio matches the expected text using Gemini API.
    Returns True if the audio matches the text, False otherwise.
    """
    prompt = f"Does the audio say the following text: '{expected_text}'? Answer only with 'Yes' or 'No'."
    audio_result = gemini.generate_from_videos([audio_path], prompt)
    
    # Check if the result contains "Yes"
    return "yes" in audio_result.lower()

# Function to generate audio with verification and retry logic
def generate_verified_audio(cosyvoice, text, emotion, voice_prompt, output_path):
    """
    Generate audio with verification and retry logic.
    Returns True if generation was successful, False otherwise.
    Also saves failed attempts with suffixes.
    """
    # Parse base filename for failed versions
    base_path, ext = os.path.splitext(output_path)
    
    # First attempt with specified emotion
    formatted_emotion = f"#{emotion}#"
    for j, result in enumerate(cosyvoice.inference_instruct2(text, formatted_emotion, voice_prompt, stream=False)):
        torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
        
        # Verify audio
        if verify_audio_matches_text(output_path, text):
            print(f"✓ Audio verification passed for: '{text}'")
            return True
        
        # First retry with the same parameters - save with failed1 suffix
        print(f"✗ Audio verification failed, retrying with same parameters for: '{text}'")
        failed1_path = f"{base_path}_failed1{ext}"
        for j, result in enumerate(cosyvoice.inference_instruct2(text, formatted_emotion, voice_prompt, stream=False)):
            torchaudio.save(failed1_path, result['tts_speech'], cosyvoice.sample_rate)
            
            if verify_audio_matches_text(failed1_path, text):
                print(f"✓ Audio verification passed on retry for: '{text}'")
                # Copy successful attempt to original path
                torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
                return True
            
            # Second retry with empty emotion - save with failed2 suffix
            print(f"✗ Audio verification failed again, retrying with empty emotion for: '{text}'")
            empty_emotion = ""
            failed2_path = f"{base_path}_failed2{ext}"
            for j, result in enumerate(cosyvoice.inference_instruct2(text, empty_emotion, voice_prompt, stream=False)):
                torchaudio.save(failed2_path, result['tts_speech'], cosyvoice.sample_rate)
                
                if verify_audio_matches_text(failed2_path, text):
                    print(f"✓ Audio verification passed with empty emotion for: '{text}'")
                    # Copy successful attempt to original path
                    torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
                    return True
                else:
                    print(f"✗ Audio verification failed with all attempts for: '{text}'")
                    # Keep all failed attempts on disk for inspection
                    return False
    
    return False

# Counter for generating unique filenames
segment_counter = 0

# Store all generated audio files information for proper ordering
audio_segments = []

# Process each line from the script
for line_idx, line in enumerate(voice_lines):
    segment_counter = 0
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
            emotion, voice_type, clean_text = extract_emotion_voice_and_clean(segment_text)
            
            if clean_text:  # Only process if there's actual content
                # Select the appropriate voice prompt
                prompt_speech = voice_prompts.get(voice_type, default_prompt_speech_16k)
                
                # Generate voice for this segment with verification
                output_path = f'Pipeline/voices/line_{line_idx}_{segment_type}_{segment_counter}.wav'
                success = generate_verified_audio(cosyvoice, clean_text, emotion, prompt_speech, output_path)
                
                if success:
                    print(f"Saved {segment_type} for line {line_idx}: '{clean_text}' with emotion '{emotion}' and voice '{voice_type}'")
                    
                    # Store information for ordering
                    audio_segments.append({
                        'line': line_idx,
                        'segment': segment_counter,
                        'type': segment_type,
                        'path': output_path
                    })
                    
                    segment_counter += 1
                else:
                    print(f"Failed to generate satisfactory audio for: '{clean_text}'")
    else:
        # No dialogue tags, process as single narration
        emotion, voice_type, clean_text = extract_emotion_voice_and_clean(line)
        
        # Select the appropriate voice prompt
        prompt_speech = voice_prompts.get(voice_type, default_prompt_speech_16k)
        
        # Generate voice for this line with verification
        output_path = f'Pipeline/voices/line_{line_idx}_narration_{segment_counter}.wav'
        success = generate_verified_audio(cosyvoice, clean_text, emotion, prompt_speech, output_path)
        
        if success:
            print(f"Saved narration for line {line_idx}: '{clean_text}' with emotion '{emotion}' and voice '{voice_type}'")
            
            # Store information for ordering
            audio_segments.append({
                'line': line_idx,
                'segment': segment_counter,
                'type': 'narration',
                'path': output_path
            })
            
            segment_counter += 1
        else:
            print(f"Failed to generate satisfactory audio for: '{clean_text}'")

print("Voice generation complete.")

# Now concatenate all audio files into one complete audio file
print("Concatenating all voice clips into a complete audio file...")

# Get all generated wav files in the correct order using our tracking
# Sort first by line, then by segment number within each line
audio_segments.sort(key=lambda x: (x['line'], x['segment']))
ordered_files = [segment['path'] for segment in audio_segments]

# Natural sort function is not needed anymore since we're already sorting correctly by the segment tracking
# We'll use the full paths directly rather than extracting basenames

# Function to load and concatenate audio files
def concatenate_audio_files(file_list, sample_rate):
    waveforms = []
    for filepath in file_list:
        waveform, sr = torchaudio.load(filepath)
        assert sr == sample_rate, f"Sample rate mismatch: {sr} vs {sample_rate}"
        waveforms.append(waveform)
    
    # Concatenate along time dimension (dim=1)
    if waveforms:
        return torch.cat(waveforms, dim=1)
    else:
        return None

# Concatenate all files and save the result
if ordered_files:
    combined_waveform = concatenate_audio_files(ordered_files, cosyvoice.sample_rate)
    if combined_waveform is not None:
        combined_output_path = os.path.join(voices_dir, 'complete_voiceover.wav')
        torchaudio.save(combined_output_path, combined_waveform, cosyvoice.sample_rate)
        print(f"Successfully saved complete voiceover to {combined_output_path}")
    else:
        print("No valid audio files found to concatenate")
else:
    print("No voice files found to concatenate")

# Optional: Create separate scene-based audio files
scene_clips = {}
for segment in audio_segments:
    # Extract line number directly from the segment data
    line_num = segment['line']
    
    # Find which scene this line belongs to
    scene_num = None
    line_count = 0
    for i, scene in enumerate(script_data['script']):
        line_count_in_scene = len(scene['clips'])
        if line_num < line_count + line_count_in_scene:
            scene_num = i + 1  # Scene numbering starts at 1
            break
        line_count += line_count_in_scene
    
    if scene_num:
        if scene_num not in scene_clips:
            scene_clips[scene_num] = []
        scene_clips[scene_num].append(segment['path'])

# Concatenate and save audio for each scene
for scene_num, files in scene_clips.items():
    # The files are already in the correct order from audio_segments
    scene_waveform = concatenate_audio_files(files, cosyvoice.sample_rate)
    if scene_waveform is not None:
        scene_output_path = os.path.join(voices_dir, f'scene_{scene_num}_voiceover.wav')
        torchaudio.save(scene_output_path, scene_waveform, cosyvoice.sample_rate)
        print(f"Saved voiceover for Scene {scene_num} to {scene_output_path}")

print("Audio concatenation complete.")

gemini = GeminiAPI(
    api_key="your_gemini_key_here",
    proxy='your_proxy_here'
)
audio_paths = ["Pipeline/voices/complete_voiceover.wav"]
audio_result = gemini.generate_from_videos(audio_paths, "prompt")
