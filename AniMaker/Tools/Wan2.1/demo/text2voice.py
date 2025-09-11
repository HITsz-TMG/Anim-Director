import os
import sys
sys.path.append('Tools/CosyVoice')
sys.path.append('Tools/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('Tools/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

voice_prompt_dir='Tools/CosyVoice/asset/speech'
voice_prompts = {
    "boy": load_wav(os.path.join(voice_prompt_dir, 'boy.wav'), 16000),
    "girl": load_wav(os.path.join(voice_prompt_dir, 'girl.mp3'), 16000),
    "youngman": load_wav(os.path.join(voice_prompt_dir, 'youngman.mp3'), 16000),
    "youngwoman": load_wav(os.path.join(voice_prompt_dir, 'youngwoman.mp3'), 16000),
    "man": load_wav(os.path.join(voice_prompt_dir, 'man.mp3'), 16000),
    "woman": load_wav(os.path.join(voice_prompt_dir, 'woman.wav'), 16000),
    "elderly": load_wav(os.path.join(voice_prompt_dir, 'elderly.mp3'), 16000),
    "narrative": load_wav(os.path.join(voice_prompt_dir, 'narrative_low.mp3'), 16000)
}

text = "Finished drawing, Bunny proudly holds up his artwork."
formatted_emotion = ""
output_path = "Tools/Wan2.1/demo/outputs/85-3-3.mp3"

for j, result in enumerate(cosyvoice.inference_instruct2(text, formatted_emotion, voice_prompts['narrative'], stream=False)):
            torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)