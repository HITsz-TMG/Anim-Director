import argparse
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import edit_distance
# from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import time
import logging
from deepface import DeepFace
import tempfile


##### celebrity id score #####
def calculate_celebrity_id_score(video_path, img_paths):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video
    frames = []
    face_verify_score_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)
    cap.release()

    face_images = img_paths
    # Save the frame as a temporary file
    for i in range(len(frames)):
        frame = frames[i]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame_file:
            frame_pil = Image.fromarray(frame)
            frame_pil.save(temp_frame_file.name)
            temp_frame_file.flush()  # Make sure the file is written to disk

        distance = []
        for j in range(len(face_images)): 
            face_gt = face_images[j]
            # Calculate the distance using DeepFace.verify() with the temporary file paths
            distance.append(DeepFace.verify(img1_path=face_gt, img2_path=temp_frame_file.name, enforce_detection = False)['distance'])
        face_verify_score_frames.append(min(distance))
        # Delete the temp_frame_file from the local disk
        os.remove(temp_frame_file.name)
    face_verify_score_frames = np.array(face_verify_score_frames)
    face_verify_score_avg = np.mean(face_verify_score_frames).item()
    
    return face_verify_score_avg

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='celebrity_id_score', help="Specify the metric to be used")
    args = parser.parse_args()

    dir_videos = args.dir_videos
    metric = args.metric

    dir_path_face = './celebrities/'
    dir_prompts =  '../../prompts/'
   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
    prompt_paths = [os.path.join(dir_prompts, os.path.splitext(os.path.basename(x))[0]+'.txt') for x in video_paths]

    # Load pretrained models
    device =  "cpu"
    import json
    # Load the JSON data from the file
    with open("../../metadata.json", "r") as infile:
        data = json.load(infile)
    # Extract the dictionaries
    face_vid = {}
    text_vid = {}
    color_vid = {}
    count_vid = {}
    amp_vid = {}
    action_vid = {}
    for item_key, item_value in data.items():
        attributes = item_value["attributes"]
        face = attributes.get("face", "")
        text = attributes.get("text", "")
        color = attributes.get("color", "")
        count = attributes.get("count", "")
        amp = attributes.get("amp", "")
        action = attributes.get("action", "")
        if face:
            face_vid[item_key] = face
        if text:
            text_vid[item_key] = text
        if color:
            color_vid[item_key] = color
        if count:
            count_vid[item_key] = count
        if amp:
            amp_vid[item_key] = amp
        if action:
            action_vid[item_key] = action

    image_paths = {}

    for key, name in face_vid.items():
        image_paths[key] = [f"{dir_path_face}{name.replace(' ', '_')}_{i}.jpg" for i in range(1, 4)]

    
    # Calculate SD scores for all video-text pairs
    scores = []
    # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"../../../results", exist_ok=True)
    # Set up logging
    log_file_path = f"../../../results/{metric}_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"../../../results/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        prompt_path = prompt_paths[i]
        basename = os.path.basename(video_path)[:4]
        if  basename in face_vid.keys():
            score = calculate_celebrity_id_score(video_path, image_paths[basename])
        else:
            score = None

        if score is not None:
            scores.append(score)
            average_score = sum(scores) / len(scores)
            logging.info(f"Vid: {os.path.basename(prompt_paths[i])},  Current {metric}: {score}, Current avg. {metric}: {average_score}.")

            
    # Calculate the average SD score across all video-text pairs
    average_score = sum(scores) / len(scores)
    logging.info(f"Final average {metric}: {average_score}, Total videos: {len(scores)}")

