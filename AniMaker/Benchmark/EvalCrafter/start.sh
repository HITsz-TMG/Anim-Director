export CUDA_VISIBLE_DEVICES=0

EC_path="Benchmark/EvalCrafter"
dir_videos="Benchmark/EvalCrafter/videos"

# [need for speicific platform] pip install spatial_correlation_sampler
# pip install spatial_correlation_sampler==0.4.0

# Celebrity ID Score
# cd $EC_path
# cd ./metrics/deepface
# python3 celebrity_id_score.py --dir_videos  $dir_videos

# IS
cd $EC_path
cd ./metrics
python3 is.py --dir_videos $dir_videos 

# # OCR Score
# cd $EC_path
# cd ./metrics
# python3 ocr_score.py --dir_videos $dir_videos --metric 'ocr_score'

# # VQA_A and VQA_T
cd $EC_path
cd ./metrics/DOVER
python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos


# CLIP-Score 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_score'

# Face Consistency 
# cd $EC_path
# cd ./metrics/Scores_with_CLIP 
# python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'face_consistency_score'

# SD-Score 
# cd $EC_path
# cd ./metrics/Scores_with_CLIP 
# python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'sd_score'

# BLIP-BLUE 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'blip_bleu'

# CLIP-Temp 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_temp_score'

# # # Action Score
cd $EC_path
cd ./metrics/mmaction2/demo
python3 action_score.py --dir_videos $dir_videos --metric 'action_score'


# Flow-Score
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'flow_score'

# Motion AC-Score
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'motion_ac_score'

# Warping Error
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'warping_error' 


# Count-Score
cd $EC_path
cd ./metrics/Segment-and-Track-Anything
python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'count_score'

# # Color-Score
# cd $EC_path
# cd ./metrics/Segment-and-Track-Anything
# python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'color_score' 

# Detection-Score
cd $EC_path
cd ./metrics/Segment-and-Track-Anything
python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'detection_score'


# # Final results
cd $EC_path
python eval_from_metrics.py 


