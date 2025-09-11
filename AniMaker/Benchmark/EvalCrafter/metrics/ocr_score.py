# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple
import os
import cv2
import numpy as np
import pandas as pd
import csv
import json
import pickle
import sys
from tqdm import tqdm
from nltk import edit_distance
import fastwer
from cer import calculate_cer
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
import time
import logging
# import wandb

##### Text Recognition Score #####
def cal_acc(gt_txt, pred_txt):
    # Calculate the Acc:
    ned, cer, wer = 0, 0, 0
    pred = ''
    gt = gt_txt        
    for idx in range(len(pred_txt)):
        res = pred_txt[idx]
        for line in res:
            pred += str(line[1][0])
    print(pred,'\n',gt)
    distance = edit_distance(gt, pred)
    # Follow ICDAR 2019 definition of N.E.D.
    ned += distance / max(len(pred), len(gt))
    # Obtain Sentence-Level Character Error Rate (CER)
    cer += calculate_cer(pred.split(" "), gt.split(" "))
    # Obtain Sentence-Level Word Error Rate (WER)
    wer += fastwer.score_sent(pred, gt)

    # return [100*ned, 100*cer, 100*wer]
    return [ned, cer, wer/100]


def calculate_ocr_score(video_path, gt_txt):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video 
    frames = []
    pred_txts = []
    ned, cer, wer = [], [], []
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # can't import torch and torch related packages if use this
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)

        pred_txt = ocr.ocr(frame, cls=True) # img: img for OCR, support ndarray, img_path and list or ndarray
        # for idx in range(len(pred_txt)):
        #     res = pred_txt[idx]
        #     for line in res:
        #         print(line)
        pred_txts.append(pred_txt)
        if pred_txt[0] is not None: 
            # Calculate the counting Accuracy:
            # print(gt_txt, '\n', pred_txt)
            writing_acc = cal_acc(gt_txt, pred_txt)
            ned.append(writing_acc[0])
            cer.append(writing_acc[1])
            wer.append(writing_acc[2])
            print("NED ",  writing_acc[0])
            print("CER ",  writing_acc[1])
            print("WER ",  writing_acc[2])
            print("----------------------------")
            print("NED: ", sum(ned)/len(ned))
            print("CER: ", sum(cer)/len(cer))
            print("WER: ", sum(wer)/len(wer))
            print("Done!") 
        else:
            ned.append(0)
            cer.append(0)
            wer.append(0)
    # text_recog_scores = ((1-np.array(ned)) + (1-np.array(cer)) + (1-np.array(wer)))/3
    text_recog_scores = ((np.array(ned)) + (np.array(cer)) + (np.array(wer)))/3
    text_recog_scores_avg = text_recog_scores.mean().item()
    # Calculate normalized MAE and vid clip score
    text_recog_score_mae = (np.sum(np.abs(text_recog_scores- text_recog_scores_avg)) / np.max([np.max(np.abs(text_recog_scores- text_recog_scores_avg)), 0.001])) / len(frames)
    # text_recog_score = ((1 - text_recog_score_mae) + text_recog_scores_avg) / 2

    return text_recog_scores_avg

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
    dir_prompts =  '../prompts/'
   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
    prompt_paths = [os.path.join(dir_prompts, os.path.splitext(os.path.basename(x))[0]+'.txt') for x in video_paths]

     # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"../results", exist_ok=True)
    # Set up logging
    log_file_path = f"../results/{metric}_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"../results/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    # Load pretrained models
    device =  "cpu"
    
    import json
    # Load the JSON data from the file
    with open("../metadata.json", "r") as infile:
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

    # Calculate SD scores for all video-text pairs
    scores = []
    

    test_num = 10
    test_num = len(video_paths)
    count = 0
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        prompt_path = prompt_paths[i]
        if count == test_num:
            break
        else:
            text = read_text_file(prompt_path)
            # get the videos' basenames list action_vid  for recognition
            basename = os.path.basename(video_path)[:4]
            if  basename in text_vid.keys():
                score = calculate_ocr_score(video_path, text_vid[os.path.basename(video_path)[:4]])
            else:
                score = None
            if score is not None:
                scores.append(score)
                count+=1
                average_score = sum(scores) / len(scores)
                logging.info(f"Vid: {os.path.basename(video_path)},  Current {metric}: {score}, Current avg. {metric}: {average_score} ")
                # wandb.log({
                #     f"Current {metric}": score,
                #     f"Average {metric}": average_score,
                # })
            
    # Calculate the average SD score across all video-text pairs
    average_score = sum(scores) / len(scores)
    logging.info(f"Final average {metric}: {average_score}, Total videos: {len(scores)}")

