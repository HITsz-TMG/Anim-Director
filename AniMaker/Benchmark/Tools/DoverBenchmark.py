import os
import sys
sys.path.append("Benchmark")
import torch
import yaml
import numpy as np
from pathlib import Path
import decord

# Import DOVER components
from EvalCrafter.metrics.DOVER.dover.datasets import (
    UnifiedFrameSampler, 
    ViewDecompositionDataset
)
from EvalCrafter.metrics.DOVER.dover.models import DOVER

class DoverEvaluator:
    """Evaluator class for video quality assessment using DOVER metrics"""
    
    def __init__(self, config_path=None, device="cuda"):
        """Initialize the DOVER evaluator with the specified configuration"""
        self.device = device
        
        # Default config path if not provided
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "../EvalCrafter/metrics/DOVER/dover.yml"
            )
        
        # Load configuration
        with open(config_path, "r") as f:
            self.opt = yaml.safe_load(f)
            
        # Initialize model
        self.model = DOVER(**self.opt["model"]["args"]).to(self.device)
        self.model.load_state_dict(
            torch.load(self.opt["test_load_path"], map_location=self.device)
        )
        self.model.eval()
        
        # Get dataset options
        self.dopt = self.opt["data"]["val-l1080p"]["args"]
        self.sample_types = ["aesthetic", "technical"]
        
    def evaluate_video(self, video_path):
        """Evaluate a single video using DOVER metrics"""
        # Configure dataset options for single video
        dopt = self.dopt.copy()
        dopt["anno_file"] = None
        dopt["data_prefix"] = os.path.dirname(video_path)
        
        # Get the video filename to match in the dataset
        target_video_name = os.path.basename(video_path)
        
        # Create dataset and dataloader
        dataset = ViewDecompositionDataset(dopt)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=1, pin_memory=True,
        )
        
        # Process the video
        for data in dataloader:
            if len(data.keys()) == 1:
                # Failed data
                continue
                
            # Check if this is our target video
            current_video_name = data["name"][0].split("/")[-1]
            if current_video_name != target_video_name:
                continue
            
            video = {}
            for key in self.sample_types:
                if key in data:
                    video[key] = data[key].to(self.device)
                    b, c, t, h, w = video[key].shape
                    video[key] = (
                        video[key]
                        .reshape(
                            b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                        )
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(
                            b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                        )
                    )
            
            # Run model with reduce_scores=False as in the reference code
            with torch.no_grad():
                results = self.model(video, reduce_scores=False)
                results = [np.mean(l.cpu().numpy()) for l in results]
            
            # Calculate fused score
            fused_score = self.fuse_results(results)
            
            return {
                "aesthetic": fused_score["aesthetic"] * 100,
                "technical": fused_score["technical"] * 100,
                "overall": fused_score["overall"] * 100
            }
        
        # If we get here, the video wasn't found or processed
        return None
    
    def fuse_results(self, results):
        """Fuse aesthetic and technical scores into an overall score"""
        # Use mean and std from generated videos as in the reference code
        t, a = (results[1] + 0.0758) / 0.0129, (results[0] - 0.1253) / 0.0318
        x = t * 0.6104 + a * 0.3896
        
        return {
            "aesthetic": 1 / (1 + np.exp(-a)),
            "technical": 1 / (1 + np.exp(-t)),
            "overall": 1 / (1 + np.exp(-x)),
        }
    
    def _delete_models(self):
        """Delete the model and clear CUDA memory to free up resources"""
        if hasattr(self, 'model'):
            del self.model
            print("Successfully deleted Dover Model")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class DoverBenchmark:
    """Simplified interface for video quality assessment using DOVER metrics"""
    
    def __init__(self, config_path=None, device="cuda"):
        """
        Initialize the benchmark for video evaluation
        
        Args:
            config_path (str, optional): Path to DOVER config file
            device (str, optional): Device to run evaluation on
        """
        self.evaluator = DoverEvaluator(config_path=config_path, device=device)
        self._results = None
        self.video_path = None
    
    def evaluate(self, video_path):
        """Run the evaluation and return the results"""
        self.video_path = video_path
        self._results = self.evaluator.evaluate_video(video_path)
        # Delete model immediately after evaluation to free memory
        #self.evaluator._delete_models()
        return self._results
    
    def get_aesthetic_score(self):
        """Get the aesthetic quality score (VQA_A)"""
        if self._results is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        return self._results["aesthetic"]
    
    def get_technical_score(self):
        """Get the technical quality score (VQA_T)"""
        if self._results is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        return self._results["technical"]
    
    def get_overall_score(self):
        """Get the overall quality score"""
        if self._results is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        return self._results["overall"]
    
    def print_scores(self):
        """Print all quality scores"""
        if self._results is None or self.video_path is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        print(f"Video: {Path(self.video_path).name}")
        print(f"Aesthetic Quality (VQA_A): {self._results['aesthetic']:.2f}")
        print(f"Technical Quality (VQA_T): {self._results['technical']:.2f}")
        print(f"Overall Quality: {self._results['overall']:.2f}")


# if __name__ == "__main__":
#     video_path = "Pipeline/videos_v4/scene_1/clip_2/scene_1_clip_2_3.mp4"
#     benchmark = DoverBenchmark()
#     scores = benchmark.evaluate(video_path)

#     # Use the scores in your application
#     print(f"Video quality scores: {scores}")

#     video_path = "Pipeline/videos_v4/scene_1/clip_2/scene_1_clip_2_3.mp4"
#     benchmark2 = DoverBenchmark()
#     scores = benchmark2.evaluate(video_path)

#     # Use the scores in your application
#     print(f"Video quality scores: {scores}")
