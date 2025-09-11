import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import logging
# import wandb
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torchvision.utils import save_image
# from diffusers import StableDiffusionXLPipeline
import requests
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import ipdb
# from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

def calculate_clip_score(video_path, text, model, tokenizer):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video 
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(resized_frame)

    # Convert numpy arrays to tensors, change dtype to float, and resize frames
    tensor_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames]

    # Initialize an empty tensor to store the concatenated features
    concatenated_features = torch.tensor([], device=device)

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():
        for frame in tensor_frames:
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_features = model.get_image_features(frame_input)
            concatenated_features = torch.cat((concatenated_features, frame_features), dim=0)

    # Tokenize the text
    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

    # Convert the tokenized text to a tensor and move it to the device
    text_input = text_tokens["input_ids"].to(device)

    # Generate text embeddings
    with torch.no_grad():
        text_features = model.get_text_features(text_input)

    # Calculate the cosine similarity scores
    concatenated_features = concatenated_features / concatenated_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    clip_score_frames = concatenated_features @ text_features.T
    # Calculate the average CLIP score across all frames, reflects temporal consistency 
    clip_score_frames_avg = clip_score_frames.mean().item()

    # ipdb.set_trace()

    return clip_score_frames_avg

def calculate_clip_temp_score(video_path, model):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    to_tensor = transforms.ToTensor()
    # Extract frames from the video 
    frames = []
    SD_images = []
    resize = transforms.Resize([224,224])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(frame)
    
    tensor_frames = torch.stack([resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames])

    # Get frames with interval
    # tensor_frames = torch.stack([resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames])
    # Num = 5
    # captions = []
    # # for i in range(Num):
    # N = len(tensor_frames)
    # indices = torch.linspace(0, N - 1, Num).long()
    # extracted_frames = torch.index_select(tensor_frames, 0, indices)

    # tensor_frames = [extracted_frames[i] for i in range(extracted_frames.size()[0])]
    concatenated_frame_features = []

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():  
        for frame in tensor_frames: # Too many frames in a video, must split before CLIP embedding, limited by the memory
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_feature = model.get_image_features(frame_input)
            concatenated_frame_features.append(frame_feature)

    concatenated_frame_features = torch.cat(concatenated_frame_features, dim=0)

    # Calculate the similarity scores
    clip_temp_score = []
    concatenated_frame_features = concatenated_frame_features / concatenated_frame_features.norm(p=2, dim=-1, keepdim=True)
    # ipdb.set_trace()

    for i in range(concatenated_frame_features.size()[0]-1):
        clip_temp_score.append(concatenated_frame_features[i].unsqueeze(0) @ concatenated_frame_features[i+1].unsqueeze(0).T)
    clip_temp_score=torch.cat(clip_temp_score, dim=0)
    # Calculate the average CLIP score across all frames, reflects temporal consistency 
    clip_temp_score_avg = clip_temp_score.mean().item()

    return clip_temp_score_avg

def compute_max(scorer, gt_prompts, pred_prompts):
    scores = []
    for pred_prompt in pred_prompts:
        for gt_prompt in gt_prompts:
            cand = {0: [pred_prompt]}
            ref = {0: [gt_prompt]}
            score, _ = scorer.compute_score(ref, cand)
            scores.append(score)
    return np.max(scores)

def calculate_blip_bleu_score(video_path, original_text, blip2_model, blip2_processor):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    scorer_cider = Cider()
    bleu1 = Bleu(n=1)
    bleu2 = Bleu(n=2)
    bleu3 = Bleu(n=3)
    bleu4 = Bleu(n=4)

    # Extract frames from the video
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(resized_frame)

    # Convert numpy arrays to tensors, change dtype to float, and resize frames
    tensor_frames = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames])
    # Get five captions for one video
    Num = 5
    captions = []
    # for i in range(Num):
    N = len(tensor_frames)
    indices = torch.linspace(0, N - 1, Num).long()
    extracted_frames = torch.index_select(tensor_frames, 0, indices)
    for i in range(Num):
        frame = extracted_frames[i]
        inputs = blip2_processor(images=frame, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip2_model.generate(**inputs)
        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(generated_text)

    original_text = [original_text]
    cider_score = (compute_max(scorer_cider, original_text, captions))
    bleu1_score = (compute_max(bleu1, original_text, captions))
    bleu2_score = (compute_max(bleu2, original_text, captions))
    bleu3_score = (compute_max(bleu3, original_text, captions))
    bleu4_score = (compute_max(bleu4, original_text, captions))

    blip_bleu_score_caps_avg = (bleu1_score + bleu2_score + bleu3_score + bleu4_score)/4
     
    return blip_bleu_score_caps_avg

# def calculate_sd_score(video_path, text, pipe, model):
#     # Load the video
#     cap = cv2.VideoCapture(video_path)
#     to_tensor = transforms.ToTensor()
#     # Extract frames from the video 
#     frames = []
#     SD_images = []
#     Num = 5
#     resize = transforms.Resize([224,224])
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
#         frames.append(frame
#         )
#     # # Generate SD imgs directly
#     # for i in range(Num): ## Num images for every prompt
#     #     image = pipe(text, height = 512, width= 512, num_inference_steps = 20).images[0]  #!!!!! same amount of SD images, but also can be mutiple times, TODO
#     #     # Convert the image to a tensor
#     #     image = resize(to_tensor(image))
#     #     SD_images.append(image.unsqueeze(0)) 

#     #     # Save the image to the specified directory
#     #     output_dir = "/apdcephfs/share_1290939/raphaelliu/Vid_Eval/Video_Gen/prompt700-release/SDXL_Imgs"
#     #     save_image(image, os.path.join(output_dir, f"{text[:12]}_{i}.png"))
    
#     # Load SD imgs from local paths
#     for i in range(Num): ## Num images for every prompt
#         output_dir = "/apdcephfs/share_1290939/raphaelliu/Vid_Eval/Video_Gen/prompt700-release/SDXL_Imgs"
#         SD_image_path = os.path.join(output_dir, f"{text[:12]}_{i}.png")
#         if os.path.exists(SD_image_path):
#             image = Image.open(SD_image_path)
#             # Convert the image to a tensor
#             image = resize(to_tensor(image))
#             SD_images.append(image.unsqueeze(0)) 
#         else:
#             image = pipe(text, height = 512, width= 512, num_inference_steps = 20).images[0]  #!!!!! same amount of SD images, but also can be mutiple times, TODO
#             # Convert the image to a tensor
#             image = resize(to_tensor(image))
#             SD_images.append(image.unsqueeze(0)) 
#             save_image(image,SD_image_path)

#     tensor_frames = [resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames]
#     SD_images = torch.cat(SD_images, 0)

#     concatenated_frame_features = []
#     concatenated_SDImg_features = []
#     # Generate embeddings for each frame and concatenate the features
#     with torch.no_grad():  
#         for frame in tensor_frames: # Too many frames in a video, must split before CLIP embedding, limited by the memory
#             frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
#             frame_feature = model.get_image_features(frame_input)
#             concatenated_frame_features.append(frame_feature)

#         for i in range(SD_images.size()[0]):
#             img = SD_images[i].unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
#             SDImg_feature  = model.get_image_features(img)
#             concatenated_SDImg_features.append(SDImg_feature)
#     # ipdb.set_trace()
#     concatenated_frame_features = torch.cat(concatenated_frame_features, dim=0)
#     concatenated_SDImg_features = torch.cat(concatenated_SDImg_features, dim=0)

#     # For testing SD_Img-SD_Img SD score only
#     # concatenated_frame_features = concatenated_SDImg_features

#     # Calculate the similarity scores
#     # similarity_scores = concatenated_frame_features @ concatenated_SDImg_features.T

#     # # Calculate the average SD score across all frames    
#     # sd_score = similarity_scores.mean().item()

#     # Calculate the similarity scores
#     concatenated_frame_features = concatenated_frame_features / concatenated_frame_features.norm(p=2, dim=-1, keepdim=True)
#     concatenated_SDImg_features = concatenated_SDImg_features / concatenated_SDImg_features.norm(p=2, dim=-1, keepdim=True)
#     sd_score_frames = concatenated_frame_features @ concatenated_SDImg_features.T
#     # Calculate the average CLIP score across all frames, reflects temporal consistency 
#     sd_score_frames_avg = sd_score_frames.mean().item()

#     return sd_score_frames_avg

def calculate_id_consistency_score(video_path, model):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    to_tensor = transforms.ToTensor()
    # Extract frames from the video 
    frames = []
    SD_images = []
    resize = transforms.Resize([224,224])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(frame)
    
    tensor_frames = [resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames]
    concatenated_frame_features = []

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():  
        for frame in tensor_frames: # Too many frames in a video, must split before CLIP embedding, limited by the memory
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_feature = model.get_image_features(frame_input)
            concatenated_frame_features.append(frame_feature)

    concatenated_frame_features = torch.cat(concatenated_frame_features, dim=0)

    # Calculate the similarity scores
    concatenated_frame_features = concatenated_frame_features / concatenated_frame_features.norm(p=2, dim=-1, keepdim=True)
    id_consistency_score = concatenated_frame_features[1:] @ concatenated_frame_features[0].unsqueeze(0).T
    # Calculate the average CLIP score across all frames, reflects temporal consistency 
    id_consistency_score_avg = id_consistency_score.mean().item()
   
    return id_consistency_score_avg

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


if __name__ == '__main__':
    # Load the CLIP model
    # task = 'fulljourney_videos'
    # task = 'pikavideos'
    # # task = 'modelscope'
    # # task = 'zeroscope'
    # # task = 'ours'
    # metric = "clip_score"

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='pika', help="Specify the model to be evaluated") # floor33 gen2 pika zeroscope modelscope
    parser.add_argument("--metric", type=str, default='clip_score', help="Specify the metric to be used") # 
    args = parser.parse_args()

    task = args.task
    metric = args.metric

    dir_videos = f'/apdcephfs/share_1290939/raphaelliu/Vid_Eval/Video_Gen/prompt700-release/{task}'
    dir_prompts = f'/apdcephfs/share_1290939/raphaelliu/Vid_Eval/Video_Gen/prompt700-release/prompts/'
   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir('/apdcephfs/share_1290939/raphaelliu/Vid_Eval/Video_Gen/prompt700-release/'+task)]
    prompt_paths = [os.path.join(dir_prompts, os.path.splitext(os.path.basename(x))[0]+'.txt') for x in video_paths]
    # prompt_paths = prompt_paths[:2000]
    
    # prompt_paths = prompt_paths[:2000]

    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    if metric == 'blip_bleu_score': 
        blip2_processor = AutoProcessor.from_pretrained("../../checkpoints/blip2-opt-2.7b")
        blip2_model = Blip2ForConditionalGeneration.from_pretrained("../../checkpoints/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    # elif metric == 'sd_score':
    #     clip_model = CLIPModel.from_pretrained("../../checkpoints/clip-vit-base-patch32").to(device)
    #     clip_tokenizer = AutoTokenizer.from_pretrained("../../checkpoints/clip-vit-base-patch32")
    #     # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     #     "../../checkpoints/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    #     # pipe = pipe.to(device)
    #     pipe = None
    else:
        clip_model = CLIPModel.from_pretrained("../../checkpoints/clip-vit-base-patch32").to(device)
        clip_tokenizer = AutoTokenizer.from_pretrained("../../checkpoints/clip-vit-base-patch32")
    
    # Calculate SD scores for all video-text pairs
    scores = []
    scores_temp = []
    
    # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"/apdcephfs/share_1290939/raphaelliu/Vid_Eval/results/{task}/{metric+timestamp}", exist_ok=True)
    # wandb.init(project="Vid Model Eval",name=task+"_"+metric+"_"+str(timestamp), dir='/apdcephfs/share_1290939/raphaelliu/Vid_Eval/results')
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"/apdcephfs/share_1290939/raphaelliu/Vid_Eval/results/{task}/{metric+timestamp}/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

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
            # ipdb.set_trace()
            if metric == 'clip_score':
                score = calculate_clip_score(video_path, text, clip_model, clip_tokenizer)
            elif metric == 'blip_bleu_score': 
                score = calculate_blip_bleu_score(video_path, text, blip2_model, blip2_processor)
            elif metric == 'clip_temp_score':
                score = calculate_clip_temp_score(video_path,clip_model)
            elif metric == 'id_consistency_score':
                score = calculate_id_consistency_score(video_path,clip_model)
            count+=1
            scores.append(score)
            average_score = sum(scores) / len(scores)
            logging.info(f"Vid: {os.path.basename(video_path)},  Current {metric}: {score}, Current avg. {metric}: {average_score}")
            # wandb.log({
            #     f"Current {metric}": score,
            #     f"Current avg. {metric}": average_score,
            # })
            
    # Calculate the average SD score across all video-text pairs
    # average_score = sum(scores) / len(scores)
    logging.info(f"Final average {metric}: {average_score}, Total videos: {len(scores)}")

