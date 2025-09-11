import os
import sys
sys.path.append("Benchmark/EvalCrafter/metrics/Segment-and-Track-Anything")
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
import re
import logging
logging.getLogger().handlers.clear()
import time
import wandb
import argparse
import ipdb

class SamBenchmark:
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Stream handler for displaying logs in the terminal
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(stream_handler)
        
        # Configure SAM and tracking arguments
        self.sam_args = sam_args.copy()
        self.sam_args['generator_args'] = {
            'points_per_side': 30,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 200,
        }

        # For every sam_gap frames, we use SAM to find new objects and add them for tracking
        self.segtracker_args = {
            'sam_gap': 49, # the interval to run sam to segment new objects
            'min_area': 200, # minimal mask area to add a new mask as a new object
            'max_obj_num': 255, # maximal object number to track in a video
            'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
        }

        # Detection parameters
        self.box_threshold = 0.6
        self.text_threshold = 0.5
        self.box_size_threshold = 0.5
        self.reset_image = True
        
        # COCO keywords for detection
        self.keywords = ['boy', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        self.aot_args = aot_args

    def save_prediction(self, pred_mask, output_dir, file_name):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask.save(os.path.join(output_dir,file_name))
        
    def colorize_mask(self, pred_mask):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask = save_mask.convert(mode='RGB')
        return np.array(save_mask)
        
    def draw_mask(self, img, mask, alpha=0.7, id_countour=False):
        img_mask = np.zeros_like(img)
        img_mask = img
        if id_countour:
            # very slow ~ 1s per image
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[obj_ids!=0]

            for id in obj_ids:
                # Overlay color on  binary mask
                if id <= 255:
                    color = _palette[id*3:id*3+3]
                else:
                    color = [0,0,0]
                foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
                binary_mask = (mask == id)

                # Compose image
                img_mask[binary_mask] = foreground[binary_mask]

                countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
                img_mask[countours, :] = 0
        else:
            binary_mask = (mask!=0)
            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            foreground = img*(1-alpha)+colorize_mask(mask)*alpha
            img_mask[binary_mask] = foreground[binary_mask]
            img_mask[countours,:] = 0
            
        return img_mask.astype(img.dtype)

    def create_directories(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory created: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")

        return path

    def _delete_models(self):
        """Delete models and release GPU memory"""
        if hasattr(self, 'segtracker'):
            del self.segtracker
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Models deleted and GPU memory cleared")

    def video_detection(self, video_path, grounding_caption):
        """Run object detection and tracking on a video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        pred_list = []
        det_count_frames = []
        frames = []

        torch.cuda.empty_cache()
        gc.collect()
        sam_gap = self.segtracker_args['sam_gap']
        frame_idx = 0
        frame_idx_processed = 0
        segtracker = SegTracker(self.segtracker_args, self.sam_args, self.aot_args)
        self.segtracker = segtracker  # Store reference to segtracker
        segtracker.restart_tracker()
        
        Num = 5  # Set the value of Num as per your requirement

        with torch.cuda.amp.autocast():
            while cap.isOpened():
                ret, frame = cap.read()
                if (frame_idx % Num) == 0:
                    frame_idx_processed+=1
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    if frame_idx == 0:
                        pred_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, 
                                                              self.box_threshold, self.text_threshold, 
                                                              self.box_size_threshold, self.reset_image)
                        torch.cuda.empty_cache()
                        gc.collect()
                        segtracker.add_reference(frame, pred_mask)
                    elif (frame_idx_processed % (sam_gap//Num)) == 0:
                        seg_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, 
                                                             self.box_threshold, self.text_threshold, 
                                                             self.box_size_threshold, self.reset_image)
                        torch.cuda.empty_cache()
                        gc.collect()
                        track_mask = segtracker.track(frame)
                        new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                        if np.sum(new_obj_mask > 0) >  frame.shape[0] * frame.shape[1] * 0.4:
                            new_obj_mask = np.zeros_like(new_obj_mask)
                        pred_mask = track_mask + new_obj_mask
                        segtracker.add_reference(frame, pred_mask)
                    else:
                        pred_mask = segtracker.track(frame,update_memory=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    pred_list.append(pred_mask)
                    frames.append(frame)

                    obj_ids = np.unique(pred_mask)
                    obj_ids = obj_ids[obj_ids!=0]
                    det_count_frames.append(len(obj_ids))

                    print(f"Processed frame {frame_idx_processed}, obj_num {segtracker.get_obj_num()}", end='\r')
                frame_idx += 1
            cap.release()
            
        return frames, pred_list, det_count_frames 

    def detect_color_hue_based(self, hue_value):
        if hue_value < 15:
            color = "red"
        elif hue_value < 22:
            color = "orange"
        elif hue_value < 39:
            color = "yellow"
        elif hue_value < 78:
            color = "green"
        elif hue_value < 131:
            color = "blue"
        else:
            color = "red"

        return color

    def evaluate_detection_score(self, video_path, prompt_text):
        """
        Evaluate detection score for a single video
        
        Args:
            video_path: Path to the video file
            prompt_text: Text prompt associated with the video
            
        Returns:
            dict: Detection scores for each detected object and average score
        """
        scores = {}
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.logger.info(f"Evaluating detection score for video: {video_name}")
        
        detected_objects = 0
        for keyword in self.keywords:
            num = len(re.findall(r'\b' + re.escape(keyword) + r'(s|es)?\b', prompt_text, re.IGNORECASE))      
            if num > 0:
                self.logger.info(f"Detecting object: {keyword}")
                grounding_caption = keyword
                _, pred_list, det_count_frames = self.video_detection(video_path, grounding_caption)
                
                det_frames = [1 if count > 0 else 0 for count in det_count_frames]
                det_frames = np.array(det_frames)
                det_avg = np.sum(det_frames) / det_frames.shape[0]
                
                scores[keyword] = det_avg
                detected_objects += 1
                self.logger.info(f"Object: {keyword}, Detection score: {det_avg:.4f}")
        
        # Calculate average score
        if detected_objects > 0:
            scores['average'] = sum(scores.values()) / detected_objects
            self.logger.info(f"Average detection score: {scores['average']:.4f}")
        else:
            self.logger.info("No objects detected in the video")
            scores['average'] = 0.0
            
        # Delete models and release memory
        #self._delete_models()    
        return scores
    
    def evaluate_count_score(self, video_path, count_info):
        """
        Evaluate count score for a single video
        
        Args:
            video_path: Path to the video file
            count_info: String in format "N object_name" (e.g., "3 dogs")
            
        Returns:
            float: Count score (1.0 is perfect match)
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.logger.info(f"Evaluating count score for video: {video_name}")
        
        if not count_info:
            self.logger.warning("No count information provided")
            return 0.0
            
        parts = count_info.split()
        if len(parts) < 2:
            self.logger.warning(f"Invalid count information format: {count_info}")
            return 0.0
            
        try:
            gt_count = float(parts[0])
            grounding_caption = ' '.join(parts[1:])
            
            self.logger.info(f"Looking for {gt_count} {grounding_caption}")
            _, pred_list, det_count_frames = self.video_detection(video_path, grounding_caption)
            
            det_count_frames = np.array(det_count_frames).astype('float64')
            det_count_diff_frames = np.array(np.abs(det_count_frames - gt_count)) / max(1, gt_count)  # normalize
            det_count_diff_avg = np.sum(det_count_diff_frames) / det_count_diff_frames.shape[0]
            
            if det_count_diff_avg > 1:
                det_count_diff_avg = 1
                
            score = 1 - det_count_diff_avg
            self.logger.info(f"Count score: {score:.4f}")
            
            # Delete models and release memory
            #self._delete_models()
            return score
            
        except ValueError:
            self.logger.warning(f"Invalid count value: {count_info}")
            #self._delete_models()  # Still release memory on error
            return 0.0
    
    def evaluate(self, video_path, prompt_text, count_info=None):
        """
        Evaluate a single video for both detection and count metrics
        
        Args:
            video_path: Path to the video file
            prompt_text: Text prompt associated with the video
            count_info: Optional count information string ("N object_name")
            
        Returns:
            dict: Evaluation results for detection and count scores
        """
        results = {}
        
        # Evaluate detection score
        results['detection_score'] = self.evaluate_detection_score(video_path, prompt_text)
        
        # Evaluate count score if count_info is provided
        if count_info:
            results['count_score'] = self.evaluate_count_score(video_path, count_info)
        
        # Delete models and release memory
        #self._delete_models()
        return results

# # Example usage:
# if __name__ == '__main__':
#     evaluator = SamBenchmark()
#     results = evaluator.evaluate(
#         video_path="Pipeline/videos/scene_1/clip_2/scene_1_clip_2_3.mp4", 
#         prompt_text="The seedling grows taller, the distant sun rises and sets repeatedly in fast motion, and a vibrant red rose slowly blooms from the seedling, glowing warmly, the boy in a blue shirt looks up at the glowing rose and his lips move as if exclaiming", 
#         count_info="1 flower"
#     )
#     print(results)