import os
from openai import OpenAI
from typing import Optional, Dict, Any


class DeepSeekAPI:
    def __init__(self, api_key: str, proxy: Optional[str] = None):
        self.api_key = api_key
        
        # Set up proxy if provided
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate_from_text(self, prompt: str, model: str = "deepseek-reasoner") -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        return response.choices[0].message.content


# # Example usage:
# if __name__ == "__main__":
#     # Initialize the API with your key and proxy
#     deepseek = DeepSeekAPI(
#         api_key="your_deepseek_key_here",
#         proxy='your_proxy_here'  # Optional
#     )
    
#     # Example: Process text only
#     text_result = deepseek.generate_from_text("Explain quantum computing in simple terms.")
#     print("Text-only Result:")
#     print(text_result)