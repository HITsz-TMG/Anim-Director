import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append("Benchmark/Tools")
import math
import torch
from ClipBenchmark import ClipBenchmark
from DoverBenchmark import DoverBenchmark
from DSBenchmark import DSBenchmark
from IncepBenchmark import IncepBenchmark
from IsBenchmark import IsBenchmark
from MMActionBenchmark import MMActionBenchmark
#from MuDIBenchmark import MuDIBenchmark
from MusiqBenchmark import MusiqBenchmark
from RAFTBenchmark import RAFTBenchmark
from SamBenchmark import SamBenchmark
import shutil

try:
    from Tools.deepseek_api import DeepSeekAPI
except ImportError:
    from deepseek_api import DeepSeekAPI

class Evaluator:
    def __init__(self):
        self.clip_metrics = ['clip_score', 'blip_bleu', 'clip_temp_score']
        self.incep_output_dir = "Benchmark/Tools/face_test"

        # Initialize all benchmark tools
        self.ClipEval = ClipBenchmark()
        self.DoverEval = DoverBenchmark()
        self.DSEval = DSBenchmark()
        self.IncepEval = IncepBenchmark()
        self.IsEval = IsBenchmark()
        self.MMActionEval = MMActionBenchmark()
        #self.MuDIEval = MuDIBenchmark()
        self.MusiqEval = MusiqBenchmark()
        self.RAFTEval = RAFTBenchmark()
        self.SamEval = SamBenchmark()
    
    def _extract_overlap_segment(self, video_path1, video_path2):
        """Extract and concatenate the last 2.5s of first video with first 2.5s of second video"""
        import cv2
        import tempfile
        import os
        
        # Get video information
        cap1 = cv2.VideoCapture(video_path1)
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1.release()
        
        cap2 = cv2.VideoCapture(video_path2)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        cap2.release()
        
        # Create temporary directory and files
        temp_dir = tempfile.mkdtemp()
        segment1_path = os.path.join(temp_dir, 'segment1.mp4')
        segment2_path = os.path.join(temp_dir, 'segment2.mp4')
        concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
        concatenated_path = os.path.join(temp_dir, 'concatenated.mp4')
        
        # Extract last 2.5 seconds of first video
        start_time1 = max(0, (total_frames1 / fps1) - 2.5)
        os.system(f"ffmpeg -i {video_path1} -ss {start_time1} -t 2.5 -c:v libx264 -c:a aac -strict experimental {segment1_path} -y -loglevel quiet")
        
        # Extract first 2.5 seconds of second video
        os.system(f"ffmpeg -i {video_path2} -t 2.5 -c:v libx264 -c:a aac -strict experimental {segment2_path} -y -loglevel quiet")
        
        # Create concat list file
        with open(concat_list_path, 'w') as f:
            f.write(f"file '{segment1_path}'\n")
            f.write(f"file '{segment2_path}'\n")
        
        # Concatenate the videos
        os.system(f"ffmpeg -f concat -safe 0 -i {concat_list_path} -c copy {concatenated_path} -y -loglevel quiet")
        
        return concatenated_path
    
    def evaluate_pre_continuity(self, video_path, pre_video_path, character_dict, raft_amp="slow", pre_video_continuous=True):
        """Evaluate continuity between current video and previous video"""
        pre_results = {}
        
        # Test video-to-video similarity for DS
        DSVideoRes = self.DSEval.evaluate_video_video_similarity(video_path, pre_video_path, frame_interval=1)
        pre_results['DS(DreamSim)_Pre'] = DSVideoRes
        
        # Handle face consistency checks
        character_count = len(character_dict) if character_dict else 0
        if character_count == 1:
            # Just one character, standard evaluation
            character_name, char_image_path = list(character_dict.items())[0]
            IncepVideoRes = self.IncepEval.evaluate_video_video_face_similarity(
                video_path, 
                pre_video_path, 
                self.incep_output_dir,
                character_prompt=f"{character_name}"
            )
            pre_results['Face Consistency(Incep)_Pre'] = IncepVideoRes
            pre_results['Face Consistency(Incep)_Pre_value'] = IncepVideoRes.get('all_pairs', {}).get('average_distance', 30) if isinstance(IncepVideoRes, dict) else 0
        
        elif character_count > 1:
            # Multiple characters, evaluate each separately and average
            face_consistency_pre_scores = []
            face_per_character_results = {}
            
            for character_name, char_image_path in character_dict.items():
                IncepVideoRes = self.IncepEval.evaluate_video_video_face_similarity(
                    video_path, 
                    pre_video_path, 
                    self.incep_output_dir,
                    character_prompt=f"{character_name}"
                )
                if isinstance(IncepVideoRes, dict) and 'average_distance' in IncepVideoRes:
                    face_consistency_pre_scores.append(IncepVideoRes['average_distance'])
                face_per_character_results[f"{character_name}_Pre"] = IncepVideoRes
            
            pre_results['Incep_Per_Character'] = face_per_character_results
            if face_consistency_pre_scores:
                pre_results['Face Consistency(Incep)_Pre_value'] = sum(face_consistency_pre_scores) / len(face_consistency_pre_scores)
        
        # Test 5-second overlap for warping error and semantic consistency (if continuous)
        concatenated_path = self._extract_overlap_segment(pre_video_path, video_path)
        
        # Test warping error on concatenated video
        RAFTOverlapRes = self.RAFTEval.evaluate(concatenated_path, raft_amp)
        if isinstance(RAFTOverlapRes, dict) and 'warping_error' in RAFTOverlapRes:
            pre_results['Warping Error_Pre'] = RAFTOverlapRes
            pre_results['Warping Error_Pre_value'] = RAFTOverlapRes.get('warping_error', 0)
        
        # Test semantic consistency on concatenated video only if continuous
        # if pre_video_continuous:
        #     ClipOverlapRes = self.ClipEval.evaluate(concatenated_path, prompt, ['clip_temp_score'])
        #     if 'clip_temp_score' in ClipOverlapRes:
        #         pre_results['Semantic Consistency(CLIP-Temp)_Pre'] = ClipOverlapRes
        #         pre_results['Semantic Consistency(CLIP-Temp)_Pre_value'] = ClipOverlapRes.get('clip_temp_score', 0)
                
        return pre_results
    
    def evaluate_post_continuity(self, video_path, post_video_path, character_dict, raft_amp="slow", post_video_continuous=True):
        """Evaluate continuity between current video and next video"""
        post_results = {}
        
        # Test video-to-video similarity for DS
        DSVideoRes = self.DSEval.evaluate_video_video_similarity(video_path, post_video_path, frame_interval=1)
        post_results['DS(DreamSim)_Post'] = DSVideoRes
        
        # Handle face consistency checks
        character_count = len(character_dict) if character_dict else 0
        if character_count == 1:
            # Just one character, standard evaluation
            character_name, char_image_path = list(character_dict.items())[0]
            IncepVideoRes = self.IncepEval.evaluate_video_video_face_similarity(
                video_path, 
                post_video_path, 
                self.incep_output_dir,
                character_prompt=f"{character_name}"
            )
            post_results['Face Consistency(Incep)_Post'] = IncepVideoRes
            post_results['Face Consistency(Incep)_Post_value'] = IncepVideoRes.get('all_pairs', {}).get('average_distance', 30) if isinstance(IncepVideoRes, dict) else 0
        
        elif character_count > 1:
            # Multiple characters, evaluate each separately and average
            face_consistency_post_scores = []
            face_per_character_results = {}
            
            for character_name, char_image_path in character_dict.items():
                IncepVideoRes = self.IncepEval.evaluate_video_video_face_similarity(
                    video_path, 
                    post_video_path, 
                    self.incep_output_dir,
                    character_prompt=f"{character_name}"
                )
                if isinstance(IncepVideoRes, dict) and 'average_distance' in IncepVideoRes:
                    face_consistency_post_scores.append(IncepVideoRes['average_distance'])
                face_per_character_results[f"{character_name}_Post"] = IncepVideoRes
            
            post_results['Incep_Per_Character'] = face_per_character_results
            if face_consistency_post_scores:
                post_results['Face Consistency(Incep)_Post_value'] = sum(face_consistency_post_scores) / len(face_consistency_post_scores)
        
        # Test 5-second overlap for warping error and semantic consistency (if continuous)
        concatenated_path = self._extract_overlap_segment(video_path, post_video_path)
        
        # Test warping error on concatenated video
        RAFTOverlapRes = self.RAFTEval.evaluate(concatenated_path, raft_amp)
        if isinstance(RAFTOverlapRes, dict) and 'warping_error' in RAFTOverlapRes:
            post_results['Warping Error_Post'] = RAFTOverlapRes
            post_results['Warping Error_Post_value'] = RAFTOverlapRes.get('warping_error', 0)
        
        # Test semantic consistency on concatenated video only if continuous
        # if post_video_continuous:
        #     ClipOverlapRes = self.ClipEval.evaluate(concatenated_path, prompt, ['clip_temp_score'])
        #     if 'clip_temp_score' in ClipOverlapRes:
        #         post_results['Semantic Consistency(CLIP-Temp)_Post'] = ClipOverlapRes
        #         post_results['Semantic Consistency(CLIP-Temp)_Post_value'] = ClipOverlapRes.get('clip_temp_score', 0)
                
        return post_results
    
    def evaluate(self, description, image_path, video_path, mm_action, raft_amp, sam_count, character_dict=None):
        """Run all benchmarks and return simplified scores and detailed results"""

        # Generate mudi_query_dict dynamically based on character_dict
        #mudi_query_dict = {'query_name': [], 'query_path': []}
        if character_dict:
            # Create tmp directory if it doesn't exist
            tmp_dir = "Pipeline/imgs/characters/tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            # Process each character
            for character_name, char_image_path in character_dict.items():
                # Create lowercase, underscore-separated directory name
                dir_name = character_name.lower().replace(" ", "_")
                char_tmp_dir = os.path.join(tmp_dir, dir_name)
                
                # Create character-specific directory
                os.makedirs(char_tmp_dir, exist_ok=True)
                # Copy character image to the tmp directory
                dest_path = os.path.join(char_tmp_dir, os.path.basename(char_image_path))
                shutil.copy(char_image_path, dest_path)
                # Add to mudi_query_dict
                #mudi_query_dict['query_name'].append(character_name)
                #mudi_query_dict['query_path'].append(char_tmp_dir)
        
        # Count characters to determine which benchmarks to run
        character_count = len(character_dict) if character_dict else 0
        print(f"Character count: {character_count}")
        #print(f"Generated mudi_query_dict: {mudi_query_dict}")

        simplified_results = {}
        detailed_results = {}
        
        # Run all evaluations
        ClipRes = self.ClipEval.evaluate(video_path, description, self.clip_metrics)
        # Extract all required CLIP metrics
        simplified_results['Text-Video Consistency(CLIP-Score)'] = ClipRes.get('clip_score', 0)
        simplified_results['Text-Story Consistency(BLIP-BLEU)'] = ClipRes.get('blip_bleu', 0)
        simplified_results['Semantic Consistency(CLIP-Temp)'] = ClipRes.get('clip_temp_score', 0)
        detailed_results['Semantic Consistency(CLIP-Temp)'] = ClipRes
        
        DoverRes = self.DoverEval.evaluate(video_path)
        # Extract aesthetic and technical scores
        simplified_results['VQA_A(Aesthetic)'] = DoverRes.get('aesthetic', 0)
        simplified_results['VQA_T(Technical)'] = DoverRes.get('technical', 0)
        detailed_results['Dover'] = DoverRes
        
        # Check for video-to-image comparison for DS
        DSRes = self.DSEval.evaluate_video_image_similarity(video_path, image_path, frame_interval=1)
        detailed_results['DS(DreamSim)'] = DSRes
        simplified_results['DS(DreamSim)'] = DSRes
        
        IsRes = self.IsEval.evaluate(video_path)
        simplified_results['IS'] = IsRes.get('score', 0) if isinstance(IsRes, dict) else 0
        detailed_results['IS'] = IsRes
        
        MusiqRes = self.MusiqEval.evaluate(video_path)
        simplified_results['MusIQ'] = MusiqRes
        detailed_results['MusIQ'] = MusiqRes
        
        # Skip certain benchmarks if no characters
        if character_count == 0:
            print("No characters provided, skipping D&C-DS(MuDI), Face Consistency, Action Recognition, Action Strength, Motion AC-Score")
        else:
            # Run benchmarks that require at least one character
            
            # Face Consistency checks
            if character_count == 1:
                # Just one character, standard evaluation
                character_name, char_image_path = list(character_dict.items())[0]
                print(f"Evaluating face consistency for character: {character_name}")
                IncepRes = self.IncepEval.evaluate_video_image_face_similarity(
                    video_path, 
                    char_image_path, 
                    self.incep_output_dir,
                    character_prompt=f"{character_name}"
                )
                simplified_results['Face Consistency(Incep)'] = IncepRes.get('average_distance', 30) if isinstance(IncepRes, dict) else 0
                detailed_results['Face Consistency(Incep)'] = IncepRes
            
            elif character_count > 1:
                # Multiple characters, evaluate each separately and average
                print(f"Evaluating face consistency for {character_count} characters")
                face_consistency_scores = []
                
                # Store individual character results
                detailed_results['Incep_Per_Character'] = {}
                
                for character_name, char_image_path in character_dict.items():
                    print(f"Evaluating character: {character_name}")
                    IncepRes = self.IncepEval.evaluate_video_image_face_similarity(
                        video_path, 
                        char_image_path, 
                        self.incep_output_dir,
                        character_prompt=f"{character_name}"
                    )
                    
                    if isinstance(IncepRes, dict) and 'average_distance' in IncepRes:
                        face_consistency_scores.append(IncepRes['average_distance'])
                    detailed_results['Incep_Per_Character'][character_name] = IncepRes
                
                # Calculate averages
                if face_consistency_scores:
                    simplified_results['Face Consistency(Incep)'] = sum(face_consistency_scores) / len(face_consistency_scores)
            
            # MMAction evaluation
            MMActionRes = self.MMActionEval.evaluate(video_path, mm_action)
            simplified_results['Action Recognition(Action-Score)'] = MMActionRes.get('score', 0) if isinstance(MMActionRes, dict) else 0
            detailed_results['Action Recognition(Action-Score)'] = MMActionRes
            
            # RAFT evaluation
            RAFTRes = self.RAFTEval.evaluate(video_path, raft_amp)
            # Extract all required RAFT metrics
            simplified_results['Action Strength(Flow-Score)'] = RAFTRes.get('flow_score', 0) if isinstance(RAFTRes, dict) else 0
            simplified_results['Warping Error'] = RAFTRes.get('warping_error', 0) if isinstance(RAFTRes, dict) else 0
            simplified_results['Motion AC-Score'] = RAFTRes.get('motion_ac_score', 0) if isinstance(RAFTRes, dict) else 0
            detailed_results['Motion AC-Score'] = RAFTRes
        
        # Only run MuDI if we have multiple characters
        # if character_count > 1:
        #     # Use the same video path for MuDI if no specific one provided
        #     mudi_video = video_path
        #     MuDIActionRes = self.MuDIEval.evaluate(mudi_video, mudi_query_dict)
        #     # Extract first item if tuple is returned
        #     simplified_results['D&C-DS(MuDI)'] = MuDIActionRes[0] if isinstance(MuDIActionRes, tuple) else MuDIActionRes
        #     detailed_results['D&C-DS(MuDI)'] = MuDIActionRes
        
        # Handle multiple objects in SAM evaluation
        sam_count_objects = self._parse_sam_count(sam_count)
        sam_detection_scores = []
        sam_count_scores = []
        sam_detailed_results = []
        
        for obj in sam_count_objects:
            SamRes = self.SamEval.evaluate(video_path, description, obj)
            sam_detailed_results.append({"object": obj, "result": SamRes})
            
            if isinstance(SamRes, dict):
                if 'detection_score' in SamRes:
                    obj_score = SamRes['detection_score'].get('average', 0) if isinstance(SamRes['detection_score'], dict) else 0
                    sam_detection_scores.append(obj_score)
                if 'count_score' in SamRes:
                    sam_count_scores.append(SamRes['count_score'])
        
        # Calculate average SAM scores
        simplified_results['Detection-Score'] = sum(sam_detection_scores) / len(sam_detection_scores) if sam_detection_scores else 0
        simplified_results['Count-Score'] = sum(sam_count_scores) / len(sam_count_scores) if sam_count_scores else 0
        detailed_results['SAM'] = sam_detailed_results

        # ordered_keys = [
        #     'VQA_A(Aesthetic)', 'VQA_T(Technical)', 'MusIQ', 
        #     'Text-Video Consistency(CLIP-Score)', 'Text-Story Consistency(BLIP-BLEU)', 'Detection-Score', 'Count-Score', 
        #     'DS(DreamSim)', 'D&C-DS(MuDI)', 'Face Consistency(Incep)', 'Warping Error', 'Semantic Consistency(CLIP-Temp)',
        #     'Action Recognition(Action-Score)', 'Action Strength(Flow-Score)', 'Motion AC-Score'
        # ]

        ordered_keys = [
            'VQA_A(Aesthetic)', 'VQA_T(Technical)', 'MusIQ', 
            'Text-Video Consistency(CLIP-Score)', 'Text-Story Consistency(BLIP-BLEU)', 'Detection-Score', 'Count-Score', 
            'DS(DreamSim)', 'Face Consistency(Incep)', 'Warping Error', 'Semantic Consistency(CLIP-Temp)',
            'Action Recognition(Action-Score)', 'Action Strength(Flow-Score)', 'Motion AC-Score'
        ]

        # Order results based on ordered_keys
        simplified_results = {key: simplified_results[key] for key in ordered_keys if key in simplified_results}
            
        return simplified_results, detailed_results
    
    def _parse_sam_count(self, sam_count):
        """Parse the sam_count string into a list of individual objects"""
        objects = []
        
        # Check if there are multiple objects separated by commas or 'and'
        if ',' in sam_count or ' and ' in sam_count:
            # Replace 'and' with comma for consistent splitting
            normalized = sam_count.replace(' and ', ',')
            # Split by comma
            for obj in normalized.split(','):
                obj = obj.strip()
                if obj:
                    objects.append(obj)
        else:
            objects.append(sam_count)
        
        return objects
        
    def calculate_total_score(self, simplified_scores):
        """
        Calculate a total score based on normalized and weighted individual metrics.
        Excludes IS (Inception Score) as requested.
        Also includes metrics with Pre and Post suffixes.
        """
        # Define weights for each metric (adjust these based on importance)
        base_weights = {
            'VQA_A(Aesthetic)': 1.0,
            'VQA_T(Technical)': 0.5,
            'MusIQ': 0.5,
            'Text-Video Consistency(CLIP-Score)': 0.5,
            'Text-Story Consistency(BLIP-BLEU)': 1.5,
            'Detection-Score': 0.1,
            'Count-Score': 0.1,
            'DS(DreamSim)': 0.3,
            #'D&C-DS(MuDI)': 0.8,
            'Face Consistency(Incep)': 0.5,
            'Warping Error': 0.5,
            'Semantic Consistency(CLIP-Temp)': 0.8,
            'Action Recognition(Action-Score)': 2.0,
            'Action Strength(Flow-Score)': 0.2,
            'Motion AC-Score': 0.05
        }
        
        # Skip IS as requested
        if 'IS' in simplified_scores:
            del simplified_scores['IS']
        
        # Create expanded weights dict that includes _Pre and _Post variants
        weights = {}
        for metric in simplified_scores:
            # Find the base metric name by removing _Pre or _Post suffix
            base_metric = metric
            if '_Pre' in metric:
                base_metric = metric.replace('_Pre', '')
            elif '_Post' in metric:
                base_metric = metric.replace('_Post', '')
            # Assign weight based on base metric
            for weight_key in base_weights:
                if weight_key in base_metric:
                    weights[metric] = base_weights[weight_key]
                    break
        
        # Normalize each score to a 0-100 scale     
        normalized_scores = {}
        for metric, score in simplified_scores.items():
            ## Overall Video Quality
            if 'VQA_A(Aesthetic)' in metric:
                normalized = score
            elif 'VQA_T(Technical)' in metric:
                normalized = score
            elif 'MusIQ' in metric:
                normalized = max(0, min((score * 500 - 100) * 10 + 60, 100))
            ## Text-Video Alignment
            elif 'Text-Video Consistency(CLIP-Score)' in metric:
                normalized = min(100, score * 300 + 10)
            elif 'Text-Story Consistency(BLIP-BLEU)' in metric:
                normalized = min(100, score * 300 + 60)
            elif 'Detection-Score' in metric:
                normalized = max(0, score * 100)
            elif 'Count-Score' in metric:
                normalized = max(0, score * 100)
            ## Video Consistency
            elif 'DS(DreamSim)' in metric:
                normalized = max(0, 100 - (score * 200) - 10)
            # elif 'D&C-DS(MuDI)' in metric:
            #     normalized = score * 100
            elif 'Face Consistency(Incep)' in metric:
                normalized = max(0, 110 - (score * 2))
            elif 'Warping Error' in metric:
                normalized = max(0, 100 - (score * 1000) - 10)
            elif 'Semantic Consistency(CLIP-Temp)' in metric:
                if score < 0.99:
                    normalized = 0
                else:
                    normalized = score * 10000 % 100 - 10
            ## Motion Quality   
            elif 'Action Recognition(Action-Score)' in metric:
                normalized = score * 50 + 50             
            elif 'Action Strength(Flow-Score)' in metric:
                normalized = max(0, min(100, math.log10(score) + 70))
            elif 'Motion AC-Score' in metric:
                normalized = score * 100
            
            normalized_scores[metric] = normalized
        
        # Calculate weighted sum
        score_sum = 0
        weight_sum = 0
        
        # Process metrics in the order they appear in base_weights
        for base_metric in base_weights:
            # Find all variants of this metric (base, _Pre, _Post)
            for metric in normalized_scores:
                if base_metric in metric:
                    if metric in weights:
                        score_sum += normalized_scores[metric] * weights[metric]
                        weight_sum += weights[metric]
        
        # Normalize to 0-100 scale, ensuring we don't divide by zero
        total_score = score_sum / weight_sum if weight_sum > 0 else 0
        
        return total_score, normalized_scores

    def release_memory(self):
        """Release GPU memory by deleting benchmark components and clearing CUDA cache"""
        # Delete benchmark components
        if hasattr(self, 'ClipEval'):
            del self.ClipEval
        if hasattr(self, 'DoverEval'):
            del self.DoverEval
        if hasattr(self, 'DSEval'):
            del self.DSEval
        if hasattr(self, 'IncepEval'):
            del self.IncepEval
        if hasattr(self, 'IsEval'):
            del self.IsEval
        if hasattr(self, 'MMActionEval'):
            del self.MMActionEval
        # if hasattr(self, 'MuDIEval'):
        #     del self.MuDIEval
        if hasattr(self, 'MusiqEval'):
            del self.MusiqEval
        if hasattr(self, 'RAFTEval'):
            del self.RAFTEval
        if hasattr(self, 'SamEval'):
            del self.SamEval
        torch.cuda.empty_cache() 
        print("GPU memory released successfully")


# # Example usage
# if __name__ == "__main__":
#     # Example parameters
#     deacription = "The Little Prince meets the Fennec Fox. They play together, running through the rose bushes. Laughter is heard."
#     image_path = "Pipeline/videos_v4/scene_2/clip_2/ref_frame.jpg"
#     video_path = "Pipeline/videos_v4/scene_2/clip_2/scene_2_clip_2_1.mp4"
#     pre_video_path = "Pipeline/videos_v4/scene_2/clip_1/scene_2_clip_1_1.mp4"
#     post_video_path = "Pipeline/videos_v4/scene_2/clip_3/scene_2_clip_3_1.mp4"
#     mm_action = "running"  # Action to evaluate
#     raft_amp = 1.0        # RAFT motion amplitude parameter
#     sam_count = "prince, fox"  # Objects to count
#     character_dict = {
#         "Little Prince": "Pipeline/imgs/characters/Little_Prince/img.jpg",
#         "Fennec Fox": "Pipeline/imgs/characters/Fennec_Fox/img.jpg"
#     }
    
#     # Create evaluator instance
#     evaluator = Evaluator()
    
#     # EXAMPLE 1: Only evaluate the main video
#     print("\n==== EXAMPLE 1: Main Video Evaluation Only ====")
#     simplified_scores_main, detailed_results_main = evaluator.evaluate(
#         deacription, image_path, video_path, mm_action, raft_amp, sam_count, character_dict
#     )
#     # Calculate and print total score for main evaluation only
#     total_score_main, normalized_scores_main = evaluator.calculate_total_score(simplified_scores_main)
#     print(f"TOTAL SCORE (Main only): {total_score_main:.2f}/100")
    
#     # EXAMPLE 2: Evaluate main video with pre-video continuity
#     print("\n==== EXAMPLE 2: Main Video + Pre-Video Continuity ====")
#     # Evaluate pre-video continuity
#     pre_results = evaluator.evaluate_pre_continuity(
#         video_path, raft_amp, pre_video_path, character_dict, pre_video_continuous=True
#     )
#     # Extract values with _value suffix from pre_results
#     pre_simplified = {}
#     for key, value in pre_results.items():
#         if key.endswith('_value'):
#             pre_simplified[key.replace('_value', '')] = value
#         elif key.startswith('DS(DreamSim)_Pre'):
#             pre_simplified[key] = value
#     # Combine main and pre-video scores
#     combined_scores_pre = simplified_scores_main.copy()
#     combined_scores_pre.update(pre_simplified)
#     # Calculate total score for main + pre
#     total_score_pre, normalized_scores_pre = evaluator.calculate_total_score(combined_scores_pre)
#     print(f"TOTAL SCORE (Main + Pre): {total_score_pre:.2f}/100")
    
#     # EXAMPLE 3: Evaluate main video with post-video continuity
#     print("\n==== EXAMPLE 3: Main Video + Post-Video Continuity ====")
#     # Evaluate post-video continuity
#     post_results = evaluator.evaluate_post_continuity(
#         video_path, raft_amp, post_video_path, character_dict, post_video_continuous=True
#     )
#     # Extract values with _value suffix from post_results
#     post_simplified = {}
#     for key, value in post_results.items():
#         if key.endswith('_value'):
#             post_simplified[key.replace('_value', '')] = value
#         elif key.startswith('DS(DreamSim)_Pre'):
#             pre_simplified[key] = value
#     # Combine main and post-video scores
#     combined_scores_post = simplified_scores_main.copy()
#     combined_scores_post.update(post_simplified)
#     # Calculate total score for main + post
#     total_score_post, normalized_scores_post = evaluator.calculate_total_score(combined_scores_post)
#     print(f"TOTAL SCORE (Main + Post): {total_score_post:.2f}/100")
    
#     # EXAMPLE 4: Evaluate main video with both pre and post continuity
#     print("\n==== EXAMPLE 4: Main Video + Pre + Post Continuity ====")
#     # Combine all scores
#     combined_scores_all = simplified_scores_main.copy()
#     combined_scores_all.update(pre_simplified)
#     combined_scores_all.update(post_simplified)
    
#     # Order the metrics for better presentation
#     ordered_keys = [
#         'VQA_A(Aesthetic)', 'VQA_T(Technical)', 'MusIQ', 
#         'Text-Video Consistency(CLIP-Score)', 'Text-Story Consistency(BLIP-BLEU)', 
#         'Detection-Score', 'Count-Score', 
#         'DS(DreamSim)', 'DS(DreamSim)_Pre', 'DS(DreamSim)_Post',
#         #'D&C-DS(MuDI)', 
#         'Face Consistency(Incep)', 'Face Consistency(Incep)_Pre', 'Face Consistency(Incep)_Post',
#         'Warping Error', 'Warping Error_Pre', 'Warping Error_Post',
#         'Semantic Consistency(CLIP-Temp)', 'Semantic Consistency(CLIP-Temp)_Pre', 'Semantic Consistency(CLIP-Temp)_Post',
#         'Action Recognition(Action-Score)', 'Action Strength(Flow-Score)', 'Motion AC-Score'
#     ]
    
#     # Sort and print metrics in a consistent order
#     print("\nAll evaluation metrics:")
#     for key in ordered_keys:
#         if key in combined_scores_all:
#             print(f"{key}: {combined_scores_all[key]}")
    
#     # Calculate total score for complete evaluation
#     total_score_all, normalized_scores_all = evaluator.calculate_total_score(combined_scores_all)
#     print(f"\nTOTAL SCORE (Complete evaluation): {total_score_all:.2f}/100")
    
#     # Release memory
#     evaluator.release_memory()
#     del evaluator
#     torch.cuda.empty_cache()
