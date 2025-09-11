import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from PIL import Image
import datetime
import os.path as osp


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_rank():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


class MusiqBenchmark:
    def __init__(self, model_path="Benchmark/MusIQ/musiq.pth", device=None, preprocess_mode='longer'):
        # Set device for evaluation
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MUSIQ model not found at {model_path}")
            
        self.model = MUSIQ(pretrained_model_path=model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess_mode = preprocess_mode
    
    def _delete_models(self):
        """
        Delete model and free GPU memory
        """
        del self.model
        torch.cuda.empty_cache()
        
    def load_video(self, video_path):
        if os.path.isdir(video_path):
            # Load frames from directory
            frames = []
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for frame_file in frame_files:
                img_path = os.path.join(video_path, frame_file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transforms.ToTensor()(img)
                frames.append(img_tensor)
            if not frames:
                raise ValueError(f"No image frames found in {video_path}")
            return torch.stack(frames)
        else:
            # For actual video files, you might need additional libraries like moviepy or cv2
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to tensor [0,1]
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                cap.release()
                if not frames:
                    raise ValueError(f"Could not load frames from video {video_path}")
                return torch.stack(frames)
            except ImportError:
                raise ImportError("OpenCV (cv2) is required to load video files. Install with: pip install opencv-python")

    def transform(self, images, preprocess_mode=None):
        if preprocess_mode is None:
            preprocess_mode = self.preprocess_mode
            
        if preprocess_mode.startswith('shorter'):
            _, _, h, w = images.size()
            if min(h,w) > 512:
                scale = 512./min(h,w)
                images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)
                if preprocess_mode == 'shorter_centercrop':
                    images = transforms.CenterCrop(512)(images)

        elif preprocess_mode == 'longer':
            _, _, h, w = images.size()
            if max(h,w) > 512:
                scale = 512./max(h,w)
                images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)

        elif preprocess_mode == 'None':
            return images / 255.

        else:
            raise ValueError("Please recheck imaging_quality_mode")
        return images / 255.
    
    def evaluate(self, video_path, output_file=None):
        """
        Evaluate a single video and return its quality score.
        
        Args:
            video_path (str): Path to the video file or directory of frames
            output_file (str, optional): Path to save results
            
        Returns:
            float: Normalized quality score of the video (0-1)
        """
        try:
            images = self.load_video(video_path)
            images = self.transform(images)
            
            # Handle videos with no frames
            if len(images) == 0:
                print(f"Warning: No frames found in {video_path}")
                return 0.0
                
            acc_score_video = 0.
            for i in range(len(images)):
                frame = images[i].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = self.model(frame)
                acc_score_video += float(score)
            
            video_score = acc_score_video/len(images)
            video_name = os.path.basename(video_path)
            normalized_score = video_score / 100.0
            
            timestamp = get_timestamp()
            log_line = f"{timestamp} Vid: {video_name}, Score: {normalized_score}"
            print(log_line)
            
            if output_file:
                with open(output_file, 'a') as f:
                    f.write(log_line + "\n")
                    
            return normalized_score
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None
    
    def evaluate_multiple_videos(self, video_list, output_file=None):
        """
        Evaluate multiple videos and return their average score.
        
        Args:
            video_list (list): List of paths to videos or directories of frames
            output_file (str, optional): Path to save results
            
        Returns:
            tuple: (average_score, individual_video_results)
        """
        video_results = []
        total_score = 0.0
        processed_count = 0
        
        for video_path in tqdm(video_list, disable=get_rank() > 0):
            normalized_score = self.evaluate(video_path, output_file)
            if normalized_score is not None:
                total_score += normalized_score
                processed_count += 1
                current_avg = total_score / processed_count
                video_results.append({'video_path': video_path, 'video_results': normalized_score * 100})
                
                timestamp = get_timestamp()
                log_line = f"{timestamp} Current avg. score: {current_avg}"
                print(log_line)
                
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(log_line + "\n")
            
        if not video_results:
            raise ValueError("No videos were successfully processed")
            
        average_score = total_score / len(video_results)
        
        # Log final result
        timestamp = get_timestamp()
        final_line = f"{timestamp} Final average score: {average_score}, Total videos: {len(video_results)}"
        print(final_line)
        
        if output_file:
            with open(output_file, 'a') as f:
                f.write(final_line + "\n")
        
        # Free GPU memory after evaluation
        #self._delete_models()
        
        return average_score, video_results


# if __name__ == "__main__":
#     evaluator = MusiqBenchmark()
#     video_path = "Pipeline/videos/scene_1/clip_2/scene_1_clip_2_3.mp4" 
#     score = evaluator.evaluate(video_path)
#     print(f"Video quality score: {score:.4f}")
#     # Free GPU memory after evaluation
#     evaluator._delete_models()
