import os
import sys
sys.path.append("Benchmark/MuDI/detect_and_compare")
import torch
import numpy as np
from PIL import Image
import cv2
from IPython.display import clear_output
from owl_dreamsim_utils import eval_with_dreamsim, eval_with_dinov2

class MuDIBenchmark:
    def __init__(self, cache_dir='Benchmark/MuDI/detect_and_compare/models', device='cuda', evaluator_type='dreamsim'):
        self.cache_dir = cache_dir
        self.device = device
        
        if evaluator_type == 'dreamsim':
            self.evaluator = eval_with_dreamsim(cache_dir=cache_dir, device=device)
        elif evaluator_type == 'dinov2':
            self.evaluator = eval_with_dinov2(cache_dir=None, device=device)
        else:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
        
    def get_gt_matrix(self, query_dict):
        embs = query_dict['query_emb']
        n = len(embs)
        gt_matrix = torch.zeros(n, n)
        # Fill the matrix
        for i in range(n):
            for j in range(i, n):  # Only calculate for i <= j
                if i == j:
                    # Diagonal: Mean of self-similarity
                    similarity = (embs[i] / embs[i].norm(dim=-1, keepdim=True)).matmul((embs[i] / embs[i].norm(dim=-1, keepdim=True)).t())
                    gt_matrix[i, j] = similarity.mean()
                else:
                    # Off-diagonal: Mean of inter-group similarity
                    inter_similarity = (embs[i] / embs[i].norm(dim=-1, keepdim=True)).matmul((embs[j] / embs[j].norm(dim=-1, keepdim=True)).t())
                    mean_similarity = inter_similarity.mean()
                    gt_matrix[i, j] = mean_similarity
                    gt_matrix[j, i] = mean_similarity  # Assign to A[j, i] without recalculating
        gt_matrix = np.array(gt_matrix)
        return gt_matrix

    def sort_by_max(self, A):
        sorted_rows = np.zeros_like(A)
        used_rows = []

        # Iterate over each column
        for i in range(A.shape[1]):
            # Find the maximum value in the i-th column that hasn't been used yet
            max_value = -np.inf
            max_index = -1
            for j in range(A.shape[0]):
                if j not in used_rows and A[j, i] > max_value:
                    max_value = A[j, i]
                    max_index = j

            # Add the row with the maximum value to the sorted array
            sorted_rows[i] = A[max_index]
            used_rows.append(max_index)
        return sorted_rows

    def gt_distance(self, scores, gt_matrix, ord=None):
        if len(scores) != len(gt_matrix):
            print(f'Count:{len(scores)}')
            return 1.
        tmp = []
        for bbox_score in scores:
            per_bbox = []
            for ref in bbox_score:
                per_bbox.append(np.array(ref).mean())
            tmp.append(per_bbox)
        
        scores = np.array(tmp)
        scores = self.sort_by_max(scores)
        return np.linalg.norm(scores - gt_matrix, ord=ord)
    
    def extract_frames(self, video_path, sample_rate=1):
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            sample_rate: Sample every nth frame
            
        Returns:
            List of PIL Image objects
        """
        frames = []
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        
        while success:
            if count % sample_rate == 0:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(image_rgb)
                frames.append(pil_image)
            
            success, image = vidcap.read()
            count += 1
            
        vidcap.release()
        return frames
    
    def evaluate_image(self, image, query_dict, threshold=0.4):
        """Evaluate a single image against the query references"""
        # Preprocess query_dict to handle both file and folder paths
        scores = self.evaluator.score(image, query_dict, threshold=threshold, return_round=False)
        gt_matrix = self.get_gt_matrix(query_dict)
        gt_score = 1 - self.gt_distance(scores, gt_matrix, ord=2)
        return gt_score
    
    def evaluate(self, video_path, query_dict, threshold=0.4, sample_rate=1):
        """
        Evaluate a video by processing individual frames
        
        Args:
            video_path: Path to the video file
            query_dict: Dictionary with reference images information
            threshold: Threshold for the evaluator
            sample_rate: Process every nth frame
            
        Returns:
            Average score across all frames
        """
        
        frames = self.extract_frames(video_path, sample_rate)
        if not frames:
            print(f"No frames could be extracted from {video_path}")
            return 0.0, []
            
        scores = []
        for i, frame in enumerate(frames):
            # Use the processed query dictionary directly
            score = self.evaluate_image(frame, query_dict, threshold)
            scores.append(score)
            if (i+1) % 10 == 0:  # Print progress every 10 frames
                print(f"Processed {i+1}/{len(frames)} frames. Current avg score: {sum(scores)/len(scores):.4f}")
                
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Release GPU memory after evaluation
        #self._delete_models()
        
        return avg_score, scores

    def save_results(self, results, filepath):
        """Save evaluation results to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for image_name, score in results:
                f.write(f"{image_name}: {score}\n")
    
    def _delete_models(self):
        """Delete models and free GPU memory"""
        if hasattr(self, 'evaluator'):
            del self.evaluator
            self.evaluator = None
        torch.cuda.empty_cache()
        print("Models deleted and GPU memory released")


# # Example usage
# if __name__ == "__main__":
#     benchmark = MuDIBenchmark()
    
#     query_dict = {
#         'query_name': ["little prince"],
#         'query_path': ["Pipeline/imgs/characters/tmp"]
#     }
    
#     video_path = "Pipeline/videos/scene_1/clip_2/scene_1_clip_2_3.mp4"  # Replace with your video path
#     if os.path.exists(video_path):
#         avg_score, frame_scores = benchmark.evaluate(video_path, query_dict)
#         print(f"Video average score: {avg_score:.4f}")
