import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from PIL import Image
import datetime
import argparse
import os.path as osp


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_rank():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def load_video(video_path):
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


def transform(images, preprocess_mode='shorter'):
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


def technical_quality(model, video_list, device, output_file=None, **kwargs):
    if 'imaging_quality_preprocessing_mode' not in kwargs:
        preprocess_mode = 'longer'
    else:
        preprocess_mode = kwargs['imaging_quality_preprocessing_mode']
    
    video_results = []
    total_score = 0.0
    processed_count = 0
    
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        try:
            images = load_video(video_path)
            images = transform(images, preprocess_mode)
            
            # Handle videos with no frames
            if len(images) == 0:
                print(f"Warning: No frames found in {video_path}")
                continue
                
            acc_score_video = 0.
            for i in range(len(images)):
                frame = images[i].unsqueeze(0).to(device)
                with torch.no_grad():  # Add no_grad for inference
                    score = model(frame)
                acc_score_video += float(score)
            
            video_score = acc_score_video/len(images)
            video_name = os.path.basename(video_path)
            normalized_score = video_score / 100.0
            
            total_score += normalized_score
            processed_count += 1
            current_avg = total_score / processed_count
            
            timestamp = get_timestamp()
            log_line = f"{timestamp} Vid: {video_name}, Current action_score: {normalized_score}, Current avg. action_score: {current_avg}"
            print(log_line)
            
            if output_file:
                with open(output_file, 'a') as f:
                    f.write(log_line + "\n")
                    
            video_results.append({'video_path': video_path, 'video_results': video_score})
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
            
    if not video_results:
        raise ValueError("No videos were successfully processed")
        
    average_score = sum([o['video_results'] for o in video_results]) / len(video_results)
    average_score = average_score / 100.
    
    # Log final result
    timestamp = get_timestamp()
    final_line = f"{timestamp} Final average action_score: {average_score}, Total videos: {len(video_results)}"
    print(final_line)
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write(final_line + "\n")
    
    return average_score, video_results


def compute_imaging_quality(video_list, device, submodules_list, output_file=None, **kwargs):
    model_path = submodules_list['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MUSIQ model not found at {model_path}")

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Process videos with the model
    with torch.no_grad():
        all_results, video_results = technical_quality(model, video_list, device, output_file, **kwargs)

    return all_results, video_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate videos with MUSIQ')
    parser.add_argument('--dir_videos', type=str, required=True, 
                        help='Directory containing videos to evaluate')
    parser.add_argument('--output_file', type=str, 
                        default="Benchmark/results/musiq.txt",
                        help='Path to save results')
    parser.add_argument('--model_path', type=str,
                        default="Benchmark/MusIQ/musiq.pth",
                        help='Path to the MUSIQ model')
    args = parser.parse_args()
    
    # Set the target directory for evaluation
    eval_dir = args.dir_videos
    
    # Get all video paths from the directory
    video_paths = []
    if os.path.exists(eval_dir):
        items = os.listdir(eval_dir)
        for item in items:
            item_path = os.path.join(eval_dir, item)
            if os.path.isdir(item_path) or item_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(item_path)
    
    if not video_paths:
        print(f"No videos found in {eval_dir}")
        exit(1)
    
    print(f"Found {len(video_paths)} videos to evaluate")
    
    # Set device for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model path
    submodules_list = {
        'model_path': args.model_path
    }
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear previous results file if exists
    if os.path.exists(args.output_file):
        open(args.output_file, 'w').close()
    
    # Evaluate videos
    try:
        avg_score, video_results = compute_imaging_quality(video_paths, device, submodules_list, args.output_file)
        print(f"\nResults saved to {args.output_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")