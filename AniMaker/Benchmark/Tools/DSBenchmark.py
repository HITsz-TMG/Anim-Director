import sys
sys.path.append("Benchmark/MuDI/detect_and_compare/dreamsim")
from dreamsim import dreamsim
from PIL import Image
import pathlib
import cv2
import numpy as np
import torch
import os

class DSBenchmark:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = dreamsim(pretrained=True, device=device)
        
    def extract_frames(self, video_path, frame_interval=16):
        """
        Extract frames from a video file
        
        Args:
            video_path (str): Path to the video file
            frame_interval (int): Interval between frames to extract
            
        Returns:
            list: List of extracted frames as PIL Images
        """
        frames = []
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = 0
        success = True
        
        while success:
            success, frame = video.read()
            if success and frame_count % frame_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                frames.append(pil_image)
            frame_count += 1
            
        video.release()
        return frames
    
    def calculate_similarity(self, img1, img2):
        """
        Calculate similarity between two images
        
        Args:
            img1 (PIL.Image): First image
            img2 (PIL.Image): Second image
            
        Returns:
            float: Similarity score
        """
        if isinstance(img1, str):
            img1 = Image.open(img1)
        if isinstance(img2, str):
            img2 = Image.open(img2)
            
        processed_img1 = self.preprocess(img1).to(self.device)
        processed_img2 = self.preprocess(img2).to(self.device)
        
        with torch.no_grad():
            distance = self.model(processed_img1, processed_img2)
            
        return distance.item()
    
    def _delete_models(self):
        """
        Delete models to free up memory
        """
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        torch.cuda.empty_cache()
    
    def evaluate_video_image_similarity(self, video_path, image_path, frame_interval=16):
        """
        Calculate average similarity between video frames and an image
        
        Args:
            video_path (str): Path to the video file
            image_path (str): Path to the image file
            frame_interval (int): Interval between frames to extract
            
        Returns:
            float: Average similarity score
        """
        frames = self.extract_frames(video_path, frame_interval)
        reference_image = Image.open(image_path)
        
        if not frames:
            raise ValueError("No frames extracted from the video")
        
        total_score = 0.0
        for frame in frames:
            score = self.calculate_similarity(frame, reference_image)
            total_score += score
            
        average_score = total_score / len(frames)
        #self._delete_models()  # Delete models after evaluation
        return average_score
    
    def evaluate_video_video_similarity(self, video_path1, video_path2, frame_interval=16):
        """
        Calculate average similarity between frames of two videos using one-to-many mapping
        
        Args:
            video_path1 (str): Path to the first video file
            video_path2 (str): Path to the second video file
            frame_interval (int): Interval between frames to extract
            
        Returns:
            float: Average similarity score
        """
        frames1 = self.extract_frames(video_path1, frame_interval)
        frames2 = self.extract_frames(video_path2, frame_interval)
        
        if not frames1 or not frames2:
            raise ValueError("No frames extracted from one or both videos")
        
        total_score = 0.0
        # For each frame in the first video
        for frame1 in frames1:
            # Find the most similar frame in the second video
            best_score = float('inf')  # Initialize with worst possible score (highest distance)
            for frame2 in frames2:
                score = self.calculate_similarity(frame1, frame2)
                best_score = min(best_score, score)  # Lower distance = higher similarity
            
            total_score += best_score
            
        average_score = total_score / len(frames1)
        return average_score
    
    def save_result(self, result, output_path="Benchmark/results/ds.txt"):
        """
        Save the result to a file
        
        Args:
            result (float): Result to save
            output_path (str): Path to save the result
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"Average Distance: {result}\n")
        print(f"Result saved to {output_path}")

# # Example usage
# if __name__ == "__main__":
#     benchmark = DSBenchmark()

#     # video_path = "Pipeline/videos/scene_1/clip_2/scene_1_clip_2_3.mp4"
#     # image_path = "Pipeline/imgs/characters/Little_Prince/img.jpg"  # Using the first image as reference
#     # distance = benchmark.evaluate_video_image_similarity(video_path, image_path, frame_interval=1)
#     # print(distance)

#     video_path1 = "Pipeline/videos_v4/scene_1/clip_2/scene_1_clip_2_3.mp4"
#     video_path2 = "Pipeline/videos_v4/scene_1/clip_2/scene_1_clip_2_4.mp4"
#     distance = benchmark.evaluate_video_video_similarity(video_path1, video_path2, frame_interval=1)
#     print(distance)
    
#     benchmark2 = DSBenchmark()
#     video_path1 = "Pipeline/videos_v4/scene_1/clip_2/scene_1_clip_2_3.mp4"
#     video_path2 = "Pipeline/videos_v4/scene_1/clip_2/scene_1_clip_2_4.mp4"
#     distance = benchmark2.evaluate_video_video_similarity(video_path1, video_path2, frame_interval=1)
#     print(distance)

