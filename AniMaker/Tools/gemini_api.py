import os
import time
import requests
import PIL.Image
from io import BytesIO
from google import genai
from google.genai import types
from typing import List, Optional, Union, Dict, Any


class GeminiAPI:
    def __init__(self, api_keys: Union[str, List[str]], proxy: Optional[str] = None):
        # Convert single API key to list if needed
        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.current_key_index = 0
        self.max_retries = 5
        
        # Set up proxy if provided
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
        
        # Initialize client with first API key
        self.client = genai.Client(api_key=self.api_keys[0])
    
    def _switch_api_key(self):
        """Switch to the next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        print(f"Switched to API key index {self.current_key_index}")
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    raise Exception(f"Failed after {self.max_retries} attempts. Last error: {str(e)}")
                
                print(f"Attempt {retries} failed: {str(e)}. Waiting 10s before retrying...")
                time.sleep(10)
                self._switch_api_key()
        
    def generate_from_text(self, prompt: str, model: str = "gemini-2.0-flash") -> str:
        """Generate content from text-only input"""
        def _generate():
            response = self.client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        
        return self._execute_with_retry(_generate)
        
    def generate_from_images(self, image_paths: List[str], prompt: str, model: str = "gemini-2.0-flash") -> str:
        """Generate content from text and images"""
        def _generate():
            # Load all images
            pil_images = [PIL.Image.open(path) for path in image_paths]
            
            # Create content array starting with the prompt
            contents = [prompt] + pil_images
            
            # Send request to API
            response = self.client.models.generate_content(
                model=model,
                contents=contents
            )
            
            return response.text
        
        return self._execute_with_retry(_generate)
    
    def generate_from_videos(self, video_paths: List[str], prompt: str, model: str = "gemini-2.0-flash") -> str:
        """Generate content from text and videos"""
        def _generate():
            uploaded_videos = []
            contents = []

            # Load all videos
            for video_path in video_paths:
                    print(f"Uploading video: {video_path}...")
                    video_file = self.client.files.upload(file=video_path)
                    print(f"Completed upload: {video_file.uri}")
                    
                    # Wait for processing
                    while video_file.state.name == "PROCESSING":
                        print('.', end='')
                        time.sleep(1)
                        video_file = self.client.files.get(name=video_file.name)
                    
                    if video_file.state.name == "FAILED":
                        raise ValueError(f"Video processing failed for {video_path}")
                    
                    uploaded_videos.append(video_file)
                    contents.append(video_file)
            
            # End with prompt
            contents.append(prompt)
            
            # Generate content from video
            response = self.client.models.generate_content(
                model=model,
                contents=[video_file, prompt]
            )
            
            return response.text
        
        return self._execute_with_retry(_generate)
    
    def generate_multimodal(self, prompt: str, image_paths: Optional[List[str]] = None, video_paths: Optional[List[str]] = None, model: str = "gemini-2.0-flash") -> str:
        """Generate content from a mix of text, images, and videos"""
        def _generate():
            # Start with prompt
            contents = [prompt]
            
            # Add images if provided
            if image_paths:
                for img_path in image_paths:
                    contents.append(PIL.Image.open(img_path))
            
            # Add videos if provided
            uploaded_videos = []
            if video_paths:
                for video_path in video_paths:
                    print(f"Uploading video: {video_path}...")
                    video_file = self.client.files.upload(file=video_path)
                    print(f"Completed upload: {video_file.uri}")
                    
                    # Wait for processing
                    while video_file.state.name == "PROCESSING":
                        print('.', end='')
                        time.sleep(1)
                        video_file = self.client.files.get(name=video_file.name)
                    
                    if video_file.state.name == "FAILED":
                        raise ValueError(f"Video processing failed for {video_path}")
                    
                    uploaded_videos.append(video_file)
                    contents.append(video_file)
                    
            # Generate content with all inputs
            response = self.client.models.generate_content(
                model=model,
                contents=contents
            )
            
            return response.text
        
        return self._execute_with_retry(_generate)
    
    def images_text_to_image(self, prompt: str, image_paths: List[str], output_path: Optional[str] = None, model: str = "gemini-2.0-flash-exp-image-generation") -> PIL.Image.Image:
        """Generate an image from input images and text prompt"""
        def _generate():
            # Load all images
            pil_images = [PIL.Image.open(path) for path in image_paths]
            
            # Create content array with prompt and images
            contents = [prompt] + pil_images
            
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            
            # Extract image from response
            image = None
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"Response text: {part.text}")
                elif part.inline_data is not None:
                    image = PIL.Image.open(BytesIO(part.inline_data.data))
                    
                    # Save image if output path is provided
                    if output_path:
                        image.save(output_path)
                        print(f"Image saved to: {output_path}")
            
            if image is None:
                raise ValueError("No image generated in response")
                
            return image
        
        return self._execute_with_retry(_generate)


# # Example usage:
# if __name__ == "__main__":
#     # Initialize the API with multiple keys and proxy
#     gemini = GeminiAPI(
#         api_keys=["your_gemini_key_1_here", 
#                  "your_gemini_key_2_here", 
#                  "your_gemini_key_3_here"],
#         proxy='your_proxy_here'
#     )
    
#     # Uncomment and use examples as needed
#     # # Example: Process text only
#     # text_result = gemini.generate_from_text("Explain quantum computing in simple terms.")
#     # print("Text-only Result:")
#     # print(text_result)
    
#     # # Example: Process images
#     # image_paths = [
#     #     "LittlePrince/s1s2_img.png",
#     #     "LittlePrince/s1s3_img.png"
#     # ]
#     # image_result = gemini.generate_from_images(image_paths, "What do these images have in common?")
#     # print("Image Analysis Result:")
#     # print(image_result)
    
#     # # Example: Process videos
#     # video_paths = ["LittlePrince/s1s1.mp4"]
#     # video_result = gemini.generate_from_videos(video_paths, "Summarize the video.")
#     # print("Video Analysis Result:")
#     # print(video_result)
    
#     # # Example: Multimodal input
#     # multimodal_result = gemini.generate_multimodal(
#     #     prompt="Explain the content of these two images and two videos.",
#     #     image_paths=["LittlePrince/s1s2_img.png", "LittlePrince/s1s3_img.png"],
#     #     video_paths=["LittlePrince/s1s2.mp4", "LittlePrince/s1s3.mp4"]
#     # )
#     # print("Multimodal Analysis Result:")
#     # print(multimodal_result)

#     # # Example: Process audios
#     # audio_paths = ["Pipeline/voices/complete_voiceover.wav"]
#     # audio_result = gemini.generate_from_videos(audio_paths, "Speech to text.")
#     # print("Aideo Analysis Result:")
#     # print(audio_result)

#     # Example: Generate image from text and reference images
#     generated_image = gemini.images_text_to_image(
#         prompt='aspect_ratio="16:9", image generation: The little boy in image 1 met the fox in image 2 in the rose garden in image 3, and they exchanged warm greetings',
#         image_paths=["Pipeline/imgs/characters/Little_Prince/img.jpg","Pipeline/imgs/characters/Fennec_Fox/img.jpg", "Pipeline/imgs/environments/Rose_Garden_Maze/Rose_Garden_Maze.png"],
#         output_path="generated_image_with_reference.png"
#     )