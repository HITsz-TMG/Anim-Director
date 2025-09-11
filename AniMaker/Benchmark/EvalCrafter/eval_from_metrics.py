import ipdb
import os
import pandas
import numpy as np
import time
import logging

#### calculate eval scores for one model 
## Load avg scores from log files 
metrics = ['VQA_A', 'VQA_T', 'IS', 'clip_temp_score', 'warping_error', 'face_consistency_score', 'action_score', 'motion_ac_score', 'flow_score', 'clip_score', 'blip_bleu', 'sd_score', 'detection_score', 'color_score', 'count_score', 'ocr_score', 'celebrity_id_score']
# quality_metrics = ['VQA_A', 'VQA_T','IS']
# temporal_metrics = ['clip_temp_score', 'warping_error', 'face_consistency_score']
# motion_metrics = ['action_score', 'motion_ac_score', 'flow_score']
# tv_align_metrics = ['clip_score', 'blip_bleu', 'sd_score', 'detection_score', 'color_score', 'count_score', 'ocr_score', 'celebrity_id_score']
metrics_dict = {metric: float('nan') for metric in metrics}


base_dir = '../results/'
model_scores_dir = base_dir
is_file = [os.path.join(model_scores_dir, f) for f in os.listdir(model_scores_dir) if 'IS' in f]
is_file = is_file[0]
txt_scores_dir = base_dir
dover_file = base_dir + "dover" + ".csv"

txt_files = [f for f in os.listdir(txt_scores_dir) if f.endswith('.txt')]
for file_path in txt_files:
    # ipdb.set_trace()
    if 'final_result' not in file_path:
        with open(txt_scores_dir+file_path, 'r') as file:
            content = file.read()
            for metric in metrics:
                if metric in content:
                    print(metric)
                    metrics_dict[metric] = float(content.split("Final")[1].split(": ")[1].split(",")[0])
                    break
        if "IS_" in file_path:
            with open(txt_scores_dir+file_path, 'r') as file:
                content = file.read()
                # ipdb.set_trace()
                metrics_dict["IS"] = float(content.split(": ")[1].split(",")[0])

# Dover
import csv
with open(dover_file, "r") as file:
    reader = csv.reader(file)
    last_row = None
    for row in reader:
        last_row = row
    # ipdb.set_trace()
    metrics_dict["VQA_A"] = float(last_row[0].split(": ")[1])
    metrics_dict["VQA_T"] = float(last_row[1])


quality_weights = np.array([0.03004555, 0.02887537, -0.01382558])*5
quality_intercept = 0.08707462696457707*5
quality_metrics = np.array([metrics_dict["VQA_A"]/100, metrics_dict["VQA_T"]/100, metrics_dict["IS"]/100])

temporal_weights = np.array([2.92492244, 0.45475678, 0.17561504])*5
temporal_intercept = -3.42274050899774*5
temporal_metrics = np.array([metrics_dict["clip_temp_score"], 1 - metrics_dict["warping_error"], metrics_dict["face_consistency_score"]])

motion_weights = np.array([-0.01641512, -0.01340959, -0.10517075])*5
motion_intercept = 0.1297562020899355*5
motion_metrics = np.array([metrics_dict["action_score"], metrics_dict["motion_ac_score"], metrics_dict["flow_score"]/100])

t2v_align_weights = np.array([-0.0701577, 0.02561424, 0.05566109, 0.0173974, -0.020954, 0.03069167, 0.00372351, 0.22686202]) * 5
t2v_align_intercept = -0.30683181901390977
t2v_align_metrics = np.array([metrics_dict["clip_score"], metrics_dict["blip_bleu"], metrics_dict["sd_score"], metrics_dict["detection_score"], metrics_dict["color_score"], metrics_dict["count_score"], 1 - metrics_dict["ocr_score"], 1- metrics_dict["celebrity_id_score"]])

quality = np.dot(quality_weights, quality_metrics) + quality_intercept
quality *= 100
# ipdb.set_trace()
temporal = np.dot(temporal_weights, temporal_metrics) + temporal_intercept
temporal *= 100
motion = np.dot(motion_weights, motion_metrics) + motion_intercept
motion *= 100
t2v_align = np.dot(t2v_align_weights, t2v_align_metrics) + t2v_align_intercept
t2v_align *= 100

total = quality + temporal +motion + t2v_align
metrics_dict["VQA_A"]/=100
metrics_dict["VQA_T"]/=100
metrics_dict["IS"]/=100
metrics_dict["flow_score"]/=100
metrics_dict["warping_error"]*=100
metrics_dict = {key: round(value*100, 2) for key, value in metrics_dict.items()}
# metrics_dict["flow_score"]=round(metrics_dict["flow_score"]/10000, 4)
metrics_dict["warping_error"]=round(metrics_dict["warping_error"]/10000, 4)

# Create the directory if it doesn't exist
metric = 'final_result'
timestamp = time.strftime("%Y%m%d-%H%M%S")
# Set up logging
log_file_path = base_dir + f"{metric}.txt"
# Delete the log file if it exists
if os.path.exists(log_file_path):
    os.remove(log_file_path)
# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# File handler for writing logs to a file
file_handler = logging.FileHandler(filename= base_dir + f"{metric}.txt")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)
# Stream handler for displaying logs in the terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

logger.addHandler(stream_handler)
logging.info(f"Metrics: {metrics_dict}")
logging.info(f"Results: Visual Quality {quality:.2f}, Text-Video Alignment {t2v_align:.2f}, Motion Quality {motion:.2f}, Temporal Consistency {temporal:.2f}, Total {total:.0f} \n")

