"""Inception Score (IS) Benchmark for evaluating single videos"""
import sys
sys.path.append("Benchmark")
import torch
import numpy as np
from decord import VideoReader, cpu
from EvalCrafter.metrics.pytorch_gan_metrics.core import calculate_inception_score, get_inception_feature
import os
import logging
logging.getLogger().handlers.clear()

class IsBenchmark:
    def __init__(self, splits=10, device=None):
        """
        Initialize the IS benchmark calculator
        
        Args:
            splits (int): Number of splits for calculating IS
            device: Device to run inference on
        """
        self.splits = splits
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def read_video_to_np(self, video_path):
        """Read a video file and convert it to numpy array"""
        vidreader = VideoReader(video_path, ctx=cpu(0))
        vid_len = len(vidreader)

        try:
            frames = vidreader.get_batch(list(range(vid_len))).asnumpy()
        except AttributeError:
            try:
                frames = vidreader.get_batch(list(range(vid_len))).numpy()
            except AttributeError:
                frames = np.array(vidreader.get_batch(list(range(vid_len))))

        return frames
    
    @torch.no_grad()
    def calculate_score(self, video_path):
        """Calculate Inception Score for a single video"""
        features = []
        
        # Read video and convert to tensor
        print(video_path)
        video = torch.tensor(self.read_video_to_np(video_path))  # t h w c
        video = video.permute(0, 3, 1, 2).contiguous()/255.  # t c h w 
        
        # Get inception features
        _, probs = get_inception_feature(video, dims=[2048, 1008], use_torch=True)
        features.append(probs)
        
        # Calculate inception score
        inception_score, std, scores = calculate_inception_score(torch.cat(features), self.splits, use_torch=True)
        
        return inception_score, std, scores
    
    def _delete_models(self):
        """Delete models and free GPU memory"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def evaluate(self, video_path):
        """Run the benchmark on a single video and return results"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        mean, std, scores = self.calculate_score(video_path)
        
        # Free GPU memory after evaluation
        #self._delete_models()
        
        result = {
            'score': mean,
            'std': std,
            'detailed_scores': scores.tolist() if isinstance(scores, torch.Tensor) else scores,
            'video_path': video_path
        }
        
        return result
    
    def print_results(self, results):
        """Print the benchmark results"""
        print(f"Inception Score for {results['video_path']}:")
        print(f"  Score: {results['score']}")
        print(f"  Std: {results['std']}")


# if __name__ == "__main__":
#     # Example usage without argparse
#     video_path = "Pipeline/videos_v4/scene_1/clip_2/scene_1_clip_2_3.mp4"  # Replace with your video path
    
#     isbenchmark = IsBenchmark()
#     results = isbenchmark.evaluate(video_path)
#     isbenchmark.print_results(results)

