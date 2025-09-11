import os
import time
import requests
import PIL.Image
from google import genai
from google.genai import types
from typing import List, Optional, Union, Dict, Any


class GeminiR1API:
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        
        # Set up proxy if provided
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version':'v1alpha'})

        
    def generate_from_text(self, prompt: str, model: str = 'gemini-2.0-flash-thinking-exp') -> str:
        """Generate content from text-only input"""
        #print(prompt)
        response = self.client.models.generate_content(
            model=model,
            contents=prompt
        )
        #print(response)
        return response.text
    


# # Example usage:
# if __name__ == "__main__":
#     # Initialize the API with your key and proxy
#     gemini = GeminiR1API(
#         api_key="your_gemini_key_here",
#         proxy='your_proxy_here'
#     )
    
#     # Example: Process text only
#     text_result = gemini.generate_from_text("Explain quantum computing in simple terms.")
#     print("Text-only Result:")
#     print(text_result)
