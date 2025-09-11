# Copyright (c) OpenMMLab. All rights reserved.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

from mmengine import Config
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer

import torch
import numpy as np
from transformers import CLIPModel, AutoTokenizer
import logging
logging.getLogger().handlers.clear()

class MMActionBenchmark:
    def __init__(self, device=None):
        # Set device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(stream_handler)
        
        # Load models
        self.load_models()

    def load_models(self):
        # Load CLIP model
        self.logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("Benchmark/EvalCrafter/checkpoints/clip-vit-base-patch32").to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained("Benchmark/EvalCrafter/checkpoints/clip-vit-base-patch32")
        
        # Load action recognition model
        self.logger.info("Loading action recognition model...")
        config = 'Benchmark/EvalCrafter/metrics/mmaction2/configs/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py'
        checkpoint = 'Benchmark/EvalCrafter/checkpoints/VideoMAE/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth'
        cfg = Config.fromfile(config)
        self.action_model = init_recognizer(cfg, checkpoint, device=self.device)
        
        # Load labels
        label_file = 'Benchmark/EvalCrafter/metrics/mmaction2/tools/data/kinetics/label_map_k400.txt'
        self.labels = [x.strip() for x in open(label_file).readlines()]

    def get_output(self, video_path, out_filename, data_sample, fps=30, font_scale=None, 
                  font_color='white', target_resolution=None):
        """Generate visualization output for action recognition results."""
        if video_path.startswith(('http://', 'https://')):
            raise NotImplementedError

        # init visualizer
        out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'
        visualizer = ActionVisualizer()
        visualizer.dataset_meta = dict(classes=self.labels)

        text_cfg = {'colors': font_color}

        visualizer.add_datasample(
            out_filename,
            video_path,
            data_sample,
            draw_pred=True,
            draw_gt=False,
            text_cfg=text_cfg,
            fps=30,
            out_type=out_type,
            out_path=osp.join('demo', out_filename),
            target_resolution=target_resolution)

    def calculate_action_score(self, video_path, action=None, out_filename=None, target_resolution=None):
        """Calculate action recognition score for a video."""
        # Run action recognition
        pred_result = inference_recognizer(self.action_model, video_path)

        # Process predictions
        pred_scores = pred_result.pred_scores.item.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top5_label = score_sorted[:5]

        # Get top-5 results
        results = [(self.labels[k[0]], k[1]) for k in top5_label]

        self.logger.info('The top-5 labels with corresponding scores are:')
        confidence = []
        action_pred = []
        for result in results:
            self.logger.info(f'{result[0]}: {result[1]}')
            action_pred.append(result[0])
            confidence.append(result[1])

        # If no action is provided, return the top prediction
        if action is None:
            return {"top_action": action_pred[0], "top_score": confidence[0]}

        # CLIP similarity calculation
        action_pred_tokens = self.clip_tokenizer(action_pred, return_tensors="pt", padding=True, truncation=True)
        text_tokens = self.clip_tokenizer(action, return_tensors="pt", padding=True, truncation=True)
        action_pred_input = torch.tensor(action_pred_tokens["input_ids"]).to(self.device)
        text_input = torch.tensor(text_tokens["input_ids"]).to(self.device)

        with torch.no_grad():
            action_pred_features = self.clip_model.get_text_features(action_pred_input)
            text_features = self.clip_model.get_text_features(text_input)

        # Calculate the similarity scores
        action_pred_features = action_pred_features / action_pred_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        action_recog_similarities = action_pred_features @ text_features.T
        action_recog_score = torch.tensor(confidence).unsqueeze(0).float().to(self.device) @ action_recog_similarities
        score = action_recog_score[0][0].item()

        # Generate visualization if requested
        if out_filename is not None:
            if target_resolution is not None:
                if target_resolution[0] == -1:
                    assert isinstance(target_resolution[1], int)
                    assert target_resolution[1] > 0
                if target_resolution[1] == -1:
                    assert isinstance(target_resolution[0], int)
                    assert target_resolution[0] > 0
                target_resolution = tuple(target_resolution)

            self.get_output(
                video_path,
                out_filename,
                pred_result,
                fps=30,
                target_resolution=target_resolution)
        
        return {
            "score": score,
            "top_predictions": list(zip(action_pred, confidence)),
            "target_action": action
        }

    def _delete_models(self):
        """Delete models and free GPU memory."""
        self.logger.info("Deleting models to free GPU memory...")
        
        # Delete CLIP model
        if hasattr(self, 'clip_model'):
            del self.clip_model
            self.clip_model = None
        
        if hasattr(self, 'clip_tokenizer'):
            del self.clip_tokenizer
            self.clip_tokenizer = None
        
        # Delete action recognition model
        if hasattr(self, 'action_model'):
            del self.action_model
            self.action_model = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Models deleted and GPU memory freed.")

    def evaluate(self, video_path, action=None, out_filename=None):
        """Evaluate a single video for action recognition."""
        #try:
        # Calculate action score
        result = self.calculate_action_score(video_path, action, out_filename)
        
        if action:
            self.logger.info(f"Video: {os.path.basename(video_path)}, Action: {action}, Score: {result['score']}")
        else:
            self.logger.info(f"Video: {os.path.basename(video_path)}, Top action: {result['top_action']}, Score: {result['top_score']}")
        
        return result
        # finally:
        #     # Free GPU memory after evaluation
        #     self._delete_models()


# if __name__ == '__main__':
#     benchmark = MMActionBenchmark()
#     result = benchmark.evaluate("Pipeline/videos/scene_1/clip_2/scene_1_clip_2_5.mp4", action="kneeling")
#     print(result)
