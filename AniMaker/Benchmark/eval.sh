export CUDA_VISIBLE_DEVICES=0

EC_path="Benchmark/EvalCrafter"
MusIQ_path="Benchmark/MusIQ"
DS_path="Benchmark/MuDI/detect_and_compare/dreamsim"
MuDI_path="Benchmark/MuDI"
Incep_path="Benchmark/Inceptionnext"
dir_videos="Benchmark/EvalCrafter/videos"

# [need for speicific platform] pip install spatial_correlation_sampler
# pip install spatial_correlation_sampler==0.4.0

## Overall Video Quality

# VQA_A and VQA_T
cd $EC_path
cd ./metrics/DOVER
python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos

# IS
cd $EC_path
cd ./metrics
python3 is.py --dir_videos $dir_videos 

# MusIQ
cd $MusIQ_path
python musiq.py --dir_videos $dir_videos 

## Text-Video Alignment

# Text-Video Consistency (CLIP-Score) 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_score'

# Text-Story Consistency (BLIP-BLEU) 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'blip_bleu'

# Detection-Score
cd $EC_path
cd ./metrics/Segment-and-Track-Anything
python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'detection_score'

# Count-Score
cd $EC_path
cd ./metrics/Segment-and-Track-Anything
python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'count_score'


## Video Consistency

# DS DreamSim
cd $DS_path
python evaluation/eval.py

# D&C-DS MuDI
cd $MuDI_path
cd ./detect_and_compare
python demo.py

# Face Consistency ##
cd $Incep_path
python face_similarity.py

# Warping Error
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'warping_error' 

# Semantic Consistency (CLIP-Temp)
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_temp_score'


## Motion Quality

# Action Recognition (Action-Score)
cd $EC_path
cd ./metrics/mmaction2/demo
python3 action_score.py --dir_videos $dir_videos --metric 'action_score'

# Action Strength (Flow-Score)
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'flow_score'

# Motion AC-Score
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'motion_ac_score'



## Final results
cd $EC_path
python eval_from_metrics.py 


