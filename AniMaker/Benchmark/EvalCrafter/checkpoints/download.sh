#!/bin/bash

# Create necessary directories
mkdir -p ckpt
mkdir -p DOVER/pretrained_weights
mkdir -p RAFT
mkdir -p VideoMAE

# Download pretrained models

# Celebrity_ID_Score
wget https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5

# IS
wget -O pt_inception-2015-12-05-6726825d.pth "https://github.com/w86763777/pytorch-gan-metrics/releases/download/v0.1.0/pt_inception-2015-12-05-6726825d.pth"

# VQA_A and VQA_T
cd DOVER/pretrained_weights  
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth 
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth
cd ..
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
wget https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth

cd ..

# CLIP-Score, Face Consistency, CLIP-Temp, Action Score
git lfs install
git clone https://huggingface.co/openai/clip-vit-base-patch32

# SD-Score 
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

# BLIP-BLUE 
git clone https://huggingface.co/Salesforce/blip2-opt-2.7b

# Action Score
wget -O tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth

# VideoMAE
cd VideoMAE
wget https://download.openmmlab.com/mmaction/v1.0/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth
cd ..

# Flow-Score, Motion AC-Score, Warping Error
cd RAFT
wget -O models.zip https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
cd ..

# Warping Error
wget https://huggingface.co/RaphaelLiu/EvalCrafter-Models/resolve/main/FlowNet2_checkpoint.pth.tar

# Count-Score, Color-Score, Detection-Score
gdown --id '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth
wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
git clone https://huggingface.co/bert-base-uncased