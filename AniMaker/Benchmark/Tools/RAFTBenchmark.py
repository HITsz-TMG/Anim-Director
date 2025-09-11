import sys
sys.path.append('Benchmark/EvalCrafter/metrics/RAFT')
sys.path.append('Benchmark/EvalCrafter/metrics/RAFT/core')

import cv2
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

import warp_utils
import torch.nn.functional as F

class Args:
    """Simple class that mimics argparse.Namespace for RAFT model compatibility"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, key):
        return key in self.__dict__

class RAFTBenchmark:
    def __init__(self, model_path='Benchmark/EvalCrafter/checkpoints/RAFT/models/raft-things.pth', small=False, mixed_precision=False, alternate_corr=False):
        """
        Initialize the video evaluator with RAFT model
        
        Args:
            model_path: Path to the RAFT model checkpoint
            small: Use small model if True
            mixed_precision: Use mixed precision if True
            alternate_corr: Use efficient correlation implementation if True
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model arguments with proper object that supports 'in' operator
        args = Args(
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
            dropout=0.0,  # Add default values for expected attributes
            model=model_path
        )
        
        # Load RAFT model
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(model_path))
        self.model = model.module
        self.model.to(self.device)
        self.model.eval()
        self.model.args.mixed_precision = mixed_precision

    def _load_video_frames(self, video_path):
        """Load frames from a video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            frames.append(frame)
        cap.release()
        return frames

    def calculate_flow_score(self, video_path):
        """Calculate the mean optical flow for a video"""
        frames = self._load_video_frames(video_path)
        optical_flows = []

        with torch.no_grad():
            for i in range(len(frames) - 1):
                image1 = frames[i]
                image2 = frames[i + 1]

                image1 = torch.tensor(image1).permute(2,0,1).float().unsqueeze(0).to(self.device)
                image2 = torch.tensor(image2).permute(2,0,1).float().unsqueeze(0).to(self.device)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)

                # Compute the magnitude of optical flow vectors
                flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
                # Calculate the mean optical flow value for the current pair of frames
                mean_optical_flow = flow_magnitude.mean().item()
                optical_flows.append(mean_optical_flow)

        mean_optical_flow_video = np.mean(optical_flows)
        print(f"Mean optical flow for the video: {mean_optical_flow_video}")

        return mean_optical_flow_video

    def calculate_motion_ac_score(self, video_path, amp):
        """Calculate motion amplitude recognition score"""
        frames = self._load_video_frames(video_path)
        optical_flows = []

        with torch.no_grad():
            for i in range(len(frames) - 1):
                image1 = frames[i]
                image2 = frames[i + 1]

                image1 = torch.tensor(image1).permute(2,0,1).float().unsqueeze(0).to(self.device)
                image2 = torch.tensor(image2).permute(2,0,1).float().unsqueeze(0).to(self.device)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)

                # Compute the magnitude of optical flow vectors
                flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
                # Calculate the mean optical flow value for the current pair of frames
                mean_optical_flow = flow_magnitude.mean().item()
                optical_flows.append(mean_optical_flow)

        mean_optical_flow_video = np.mean(optical_flows)
        print(f"Mean optical flow for the video: {mean_optical_flow_video}")
        
        if np.abs(mean_optical_flow_video) > 5:
            amp_pred = 'large'
        else:
            amp_pred = 'slow'

        if amp_pred == amp: # may use a distance to 3?
            amp_recognition_score = 1
        else:
            amp_recognition_score = 0 

        return amp_recognition_score

    def compute_video_warping_error(self, video_path):
        """Compute video warping error using optical flow"""
        frames = self._load_video_frames(video_path)
        
        Num = len(frames)
        tensor_frames = torch.stack([torch.from_numpy(frame) for frame in frames])
        N = len(tensor_frames)
        indices = torch.linspace(0, N - 1, Num).long()
        extracted_frames = torch.index_select(tensor_frames, 0, indices)
        
        err = 0
        with torch.no_grad():
            for i in range(Num - 1):
                frame1 = extracted_frames[i]
                frame2 = extracted_frames[i + 1]

                img1 = frame1.permute(2,0,1).float().unsqueeze(0).to(self.device)/ 255.0
                img2 = frame2.permute(2,0,1).float().unsqueeze(0).to(self.device)/ 255.0

                # Downsample the images by a factor of 2
                img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
                img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)

                padder = InputPadder(img1.shape)
                img1, img2 = padder.pad(img1, img2)

                # Compute forward flow
                _, fw_flow = self.model(img1, img2, iters=20, test_mode=True)
                fw_flow = warp_utils.tensor2img(fw_flow)
                torch.cuda.empty_cache()

                # Compute backward flow
                _, bw_flow = self.model(img2, img1, iters=20, test_mode=True)
                bw_flow = warp_utils.tensor2img(bw_flow)
                torch.cuda.empty_cache()

                # Compute occlusion
                fw_occ, warp_img2 = warp_utils.detect_occlusion(bw_flow, fw_flow, img2)
                warp_img2 = torch.tensor(warp_img2).float().to(self.device)
                fw_occ = torch.tensor(fw_occ).float().to(self.device)

                # Load flow and occlusion mask
                flow = fw_flow
                occ_mask = fw_occ
                noc_mask = 1 - occ_mask

                diff = (warp_img2 - img1) * noc_mask
                diff_squared = diff ** 2
                
                # Calculate the sum and mean
                N = torch.sum(noc_mask)
                if N == 0:
                    N = diff_squared.numel()
                
                err += torch.sum(diff_squared) / N

        warping_error = err / (len(extracted_frames) - 1)
        return warping_error.item()

    def _delete_models(self):
        """Delete model and release GPU memory"""
        del self.model
        torch.cuda.empty_cache()

    def evaluate(self, video_path, amp=None):
        """
        Evaluate a video on all three metrics
        
        Args:
            video_path: Path to the video file
            amp: Amplitude value for motion_ac_score calculation (either 'slow' or 'large')
            
        Returns:
            dict: Dictionary with all evaluation results
        """
        results = {}
        
        # Calculate flow score
        results['flow_score'] = self.calculate_flow_score(video_path)
        
        # Calculate warping error
        results['warping_error'] = self.compute_video_warping_error(video_path)
        
        # Calculate motion amplitude score if amp is provided
        if amp is not None:
            amp_score = self.calculate_motion_ac_score(video_path, amp)
            results['motion_ac_score'] = amp_score
        
        # Release GPU memory after evaluation
        # self._delete_models()
        
        return results


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


# if __name__ == '__main__':
#     video_path = "Pipeline/videos/scene_1/clip_2/scene_1_clip_2_3.mp4"  # Replace with actual video path
#     amp='slow'
#     evaluator = RAFTBenchmark()

#     results = evaluator.evaluate(video_path, amp)
#     # Print results
#     print("\nEvaluation Results:")
#     print("-" * 50)
#     for metric, value in results.items():
#         print(f"{metric}: {value}")
#     print("-" * 50)
