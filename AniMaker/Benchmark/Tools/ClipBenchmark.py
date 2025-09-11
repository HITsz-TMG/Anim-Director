import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import logging
logging.getLogger().handlers.clear()
# import wandb
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torchvision.utils import save_image
from diffusers import StableDiffusionXLPipeline
import requests
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import ipdb
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

class ClipBenchmark:
    def __init__(self, model_path="Benchmark/EvalCrafter/checkpoints", device=None):
        self.model_path = model_path
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.clip_model = None
        self.clip_tokenizer = None
        self.blip2_model = None
        self.blip2_processor = None
        
        # Set up logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(stream_handler)
    
    def _load_clip_model(self):
        if self.clip_model is None or self.clip_tokenizer is None:
            self.clip_model = CLIPModel.from_pretrained(f"{self.model_path}/clip-vit-base-patch32").to(self.device)
            self.clip_tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/clip-vit-base-patch32")
    
    def _load_blip_model(self):
        if self.blip2_model is None or self.blip2_processor is None:
            self.blip2_processor = AutoProcessor.from_pretrained(f"{self.model_path}/blip2-opt-2.7b")
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                f"{self.model_path}/blip2-opt-2.7b", 
                torch_dtype=torch.float16
            ).to(self.device)
    
    def evaluate(self, video_path, text, metrics=None):
        """
        Evaluate a single video with specified metrics
        
        Args:
            video_path (str): Path to the video file
            text (str): Text prompt for the video
            metrics (list): List of metrics to compute. Options: 'clip_score', 'blip_bleu', 'clip_temp_score'
                           If None, computes all three metrics.
        
        Returns:
            dict: Dictionary with metric names as keys and scores as values
        """
        if metrics is None:
            metrics = ['clip_score', 'blip_bleu', 'clip_temp_score']
        
        results = {}
        
        for metric in metrics:
            if metric == 'clip_score':
                self._load_clip_model()
                results[metric] = self.calculate_clip_score(video_path, text)
            elif metric == 'blip_bleu':
                self._load_blip_model()
                results[metric] = self.calculate_blip_bleu(video_path, text)
            elif metric == 'clip_temp_score':
                self._load_clip_model()
                results[metric] = self.calculate_clip_temp_score(video_path)
        
        # Delete models after evaluation to free memory
        # self._delete_models()
        
        return results
    
    def _delete_models(self):
        """Delete models to free up GPU memory"""
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
        
        if self.clip_tokenizer is not None:
            del self.clip_tokenizer
            self.clip_tokenizer = None
        
        if self.blip2_model is not None:
            del self.blip2_model
            self.blip2_model = None
        
        if self.blip2_processor is not None:
            del self.blip2_processor
            self.blip2_processor = None
        
        # Force garbage collection to release memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    def calculate_clip_score(self, video_path, text):
        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Extract frames from the video 
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
            frames.append(resized_frame)

        # Convert numpy arrays to tensors, change dtype to float, and resize frames
        tensor_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames]

        # Initialize an empty tensor to store the concatenated features
        concatenated_features = torch.tensor([], device=self.device)

        # Generate embeddings for each frame and concatenate the features
        with torch.no_grad():
            for frame in tensor_frames:
                frame_input = frame.unsqueeze(0).to(self.device)  # Add batch dimension and move the frame to the device
                frame_features = self.clip_model.get_image_features(frame_input)
                concatenated_features = torch.cat((concatenated_features, frame_features), dim=0)

        # Tokenize the text
        text_tokens = self.clip_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

        # Convert the tokenized text to a tensor and move it to the device
        text_input = text_tokens["input_ids"].to(self.device)

        # Generate text embeddings
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(text_input)

        # Calculate the cosine similarity scores
        concatenated_features = concatenated_features / concatenated_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        clip_score_frames = concatenated_features @ text_features.T
        # Calculate the average CLIP score across all frames, reflects temporal consistency 
        clip_score_frames_avg = clip_score_frames.mean().item()

        return clip_score_frames_avg

    def calculate_clip_temp_score(self, video_path):
        # Load the video
        cap = cv2.VideoCapture(video_path)
        to_tensor = transforms.ToTensor()
        # Extract frames from the video 
        frames = []
        resize = transforms.Resize([224,224])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        tensor_frames = torch.stack([resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames])

        concatenated_frame_features = []

        # Generate embeddings for each frame and concatenate the features
        with torch.no_grad():  
            for frame in tensor_frames: # Too many frames in a video, must split before CLIP embedding, limited by memory
                frame_input = frame.unsqueeze(0).to(self.device)  # Add batch dimension and move the frame to the device
                frame_feature = self.clip_model.get_image_features(frame_input)
                concatenated_frame_features.append(frame_feature)

        concatenated_frame_features = torch.cat(concatenated_frame_features, dim=0)

        # Calculate the similarity scores
        clip_temp_score = []
        concatenated_frame_features = concatenated_frame_features / concatenated_frame_features.norm(p=2, dim=-1, keepdim=True)

        for i in range(concatenated_frame_features.size()[0]-1):
            clip_temp_score.append(concatenated_frame_features[i].unsqueeze(0) @ concatenated_frame_features[i+1].unsqueeze(0).T)
        clip_temp_score=torch.cat(clip_temp_score, dim=0)
        # Calculate the average CLIP score across all frames, reflects temporal consistency 
        clip_temp_score_avg = clip_temp_score.mean().item()

        return clip_temp_score_avg

    def compute_max(self, scorer, gt_prompts, pred_prompts):
        scores = []
        for pred_prompt in pred_prompts:
            for gt_prompt in gt_prompts:
                cand = {0: [pred_prompt]}
                ref = {0: [gt_prompt]}
                score, _ = scorer.compute_score(ref, cand)
                scores.append(score)
        return np.max(scores)

    def calculate_blip_bleu(self, video_path, original_text):
        # Load the video
        cap = cv2.VideoCapture(video_path)

        scorer_cider = Cider()
        bleu1 = Bleu(n=1)
        bleu2 = Bleu(n=2)
        bleu3 = Bleu(n=3)
        bleu4 = Bleu(n=4)

        # Extract frames from the video
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
            frames.append(resized_frame)

        # Convert numpy arrays to tensors, change dtype to float, and resize frames
        tensor_frames = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames])
        # Get five captions for one video
        Num = 5
        captions = []
        # for i in range(Num):
        N = len(tensor_frames)
        indices = torch.linspace(0, N - 1, Num).long()
        extracted_frames = torch.index_select(tensor_frames, 0, indices)
        for i in range(Num):
            frame = extracted_frames[i]
            inputs = self.blip2_processor(images=frame, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.blip2_model.generate(**inputs)
            generated_text = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            captions.append(generated_text)

        original_text = [original_text]
        cider_score = (self.compute_max(scorer_cider, original_text, captions))
        bleu1_score = (self.compute_max(bleu1, original_text, captions))
        bleu2_score = (self.compute_max(bleu2, original_text, captions))
        bleu3_score = (self.compute_max(bleu3, original_text, captions))
        bleu4_score = (self.compute_max(bleu4, original_text, captions))

        blip_bleu_caps_avg = (bleu1_score + bleu2_score + bleu3_score + bleu4_score)/4
         
        return blip_bleu_caps_avg

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


# # Example usage:
# if __name__ == "__main__":
#     evaluator = ClipBenchmark()
#     results = evaluator.evaluate("Pipeline/videos_v4/scene_1/clip_1/scene_1_clip_1_1.mp4",
#                                 "The Little Prince with blonde hair and blue shirt whispers, his lips moving slightly, a gentle breeze rustles the tiny green seedling.", 
#                                 metrics=['clip_score', 'blip_bleu', 'clip_temp_score'])
#     print(results)
