
import os
import time
from openai import OpenAI

class GPT:

    def __init__(self, base_url="", organization="", api_key=""):
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif organization:
            self.client = OpenAI(api_key=api_key, organization=organization)
        else:
            self.client = OpenAI(api_key=api_key)
        self.max_attempts = 10
        self.max_tokens = 4096

    def query(self, prompt, image_urls=[], model=""):
        self.model = model
        if self.model == "":
            self.model = "gpt-4o"
        
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        content = [{"type": "text", "text": prompt}]
        if image_urls:
            content.extend([{"type": "image_url", "image_url": {"url": url}} for url in image_urls])
        messages.append({"role": "user", "content": content})

        attempts = 0
        while attempts < self.max_attempts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens
                )
                if response.choices[0].message.content.strip():
                    return response.choices[0].message.content
                else:
                    print("Received an empty response. Retrying in 10 seconds.")
            except Exception as e:
                print(messages)
                print(f"Error occurred: {e}. Retrying in 10 seconds.")
                time.sleep(10)
                attempts += 1

        raise Exception("Max attempts reached, failed to get a response from OpenAI.") 
