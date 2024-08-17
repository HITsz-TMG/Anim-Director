import requests
import json
import time

# Optional operations：draw(prompt)、enlarge(origin_task_id, index)、vary(origin_task_id, index, prompt, aspect_ratio)、region(origin_task_id, prompt, mask)、zoom(origin_task_id, zoom_ratio, aspect_ratio, prompt)
class MidJourneyAPI:
    def __init__(self, api_key=""):
        self.base_url = "https://api.midjourneyapi.xyz/mj/v2/"
        self.headers = {"X-API-KEY": api_key}

    # The following are basic methods that can be executed directly   
    def fetch(self, task_id):
        endpoint = f"{self.base_url}fetch"
        data = {"task_id": task_id}
        return requests.post(endpoint, json=data)
    
    def imagine(self, prompt):
        endpoint = f"{self.base_url}imagine"
        data = {"prompt": prompt, "process_mode": "fast", "skip_prompt_check": True}
        #data = {"prompt": prompt, "process_mode": "relax", "skip_prompt_check": True} 
        return requests.post(endpoint, headers=self.headers, json=data)
    
    def upscale(self, origin_task_id, index):
        endpoint = f"{self.base_url}upscale"
        data = {"origin_task_id": origin_task_id, "index": index}
        return requests.post(endpoint, headers=self.headers, json=data)
    
    def variation(self, origin_task_id, index, prompt):
        endpoint = f"{self.base_url}variation"
        data = {"origin_task_id": origin_task_id, "index": index, "prompt": prompt, "process_mode": "fast", "skip_prompt_check": True}
        #data = {"origin_task_id": origin_task_id, "index": index, "prompt": prompt, "process_mode": "relax", "skip_prompt_check": True}
        return requests.post(endpoint, headers=self.headers, json=data)
    
    def inpaint(self, origin_task_id, prompt, mask=""):
        endpoint = f"{self.base_url}inpaint"
        data = {"origin_task_id": origin_task_id, "prompt": prompt, "mask": mask, "process_mode": "fast", "skip_prompt_check": True}
        #data = {"origin_task_id": origin_task_id, "prompt": prompt, "mask": mask, "process_mode": "relax", "skip_prompt_check": True}
        payload = json.dumps(data)
        headers = {'X-API-Key': self.headers['X-API-KEY'], 'Content-Type': 'application/json'}
        return requests.post(endpoint, headers=headers, data=payload)
    
    def outpaint(self, origin_task_id, zoom_ratio, prompt):
        endpoint = f"{self.base_url}outpaint"
        data = {"origin_task_id": origin_task_id, "zoom_ratio": zoom_ratio, "prompt": prompt}
        return requests.post(endpoint, headers=self.headers, json=data)

    # The following are methods for direct external calls
    def draw(self, prompt, max_retries=20): # Return a four-grid image. If necessary, you can get a single image through image_urls
        imagine_response = self.imagine(prompt)
        if imagine_response.status_code == 200:
            task_id = imagine_response.json().get('task_id')
            retries = 0
            while retries < max_retries:
                fetch_response = self.fetch(task_id)
                if fetch_response.status_code == 200:
                    image_url = fetch_response.json().get('task_result', {}).get('image_url')
                    if image_url:
                        return task_id, image_url
                    else:
                        time.sleep(30)
                        retries += 1
                else:
                    print(task_id)
                    raise Exception(f"Fetch failed with status {fetch_response.status_code}: {fetch_response.json()}")
            print(task_id)
            raise Exception("Max retries reached without getting an image URL.")
        else:
            raise Exception(f"Imagine failed with status {imagine_response.status_code}: {imagine_response.json()}")

    def enlarge(self, origin_task_id, index, max_retries=20): # Operate on four-grid images; return to a single image
        upscale_response = self.upscale(origin_task_id, index)
        success = upscale_response.json().get('success')
        if success == True:
            task_id = upscale_response.json().get('task_id')
            retries = 0
            while retries < max_retries:
                print(task_id)
                fetch_response = self.fetch(task_id)
                if fetch_response.status_code == 200:
                    image_url = fetch_response.json().get('task_result', {}).get('image_url')
                    if image_url:
                        return task_id, image_url
                    else:
                        time.sleep(30)
                        retries += 1
                else:
                    print(task_id)
                    raise Exception(f"Fetch failed with status {fetch_response.status_code}: {fetch_response.json()}")
            print(task_id)
            raise Exception("Max retries reached without getting an image URL.")
        else:
            print(task_id)
            raise Exception(f"Upscale failed with status {upscale_response.json()}")

    def vary(self, origin_task_id, index, prompt, max_retries=20): # Operate on a four-grid (index is a number)/single (index is high_variation/low_variation) image; return a four-grid image. If necessary, you can get a single image through image_urls.
        variation_response = self.variation(origin_task_id, index, prompt)
        if variation_response.status_code == 200:
            task_id = variation_response.json().get('task_id')
            retries = 0
            while retries < max_retries:
                fetch_response = self.fetch(task_id)
                if fetch_response.status_code == 200:
                    image_url = fetch_response.json().get('task_result', {}).get('image_url')
                    if image_url:
                        return task_id, image_url
                    else:
                        time.sleep(30)
                        retries += 1
                else:
                    raise Exception(f"Fetch failed with status {fetch_response.status_code}: {fetch_response.json()}")
            raise Exception("Max retries reached without getting an image URL.")
        else:
            raise Exception(f"Variation failed with status {variation_response.status_code}: {variation_response.json()}")

    def region(self, origin_task_id, prompt, mask, max_retries=20): # Operate on a single image; return to four-grid images对单张图片操作; If necessary, you can get a single image through image_urls.
        inpaint_response = self.inpaint(origin_task_id, prompt, mask)
        if inpaint_response.status_code == 200:
            task_id = inpaint_response.json().get('task_id')
            retries = 0
            while retries < max_retries:
                fetch_response = self.fetch(task_id)
                if fetch_response.status_code == 200:
                    image_url = fetch_response.json().get('task_result', {}).get('image_url')
                    if image_url:
                        return task_id, image_url
                    else:
                        time.sleep(30)
                        retries += 1
                else:
                    raise Exception(f"Fetch failed with status {fetch_response.status_code}: {fetch_response.json()}")
            raise Exception("Max retries reached without getting an image URL.")
        else:
            raise Exception(f"Inpaint failed with status {inpaint_response.status_code}: {inpaint_response.json()}")

    def zoom(self, origin_task_id, zoom_ratio, prompt, max_retries=20): # Operate on a single image; return to four-grid images对单张图片操作; If necessary, you can get a single image through image_urls.
        outpaint_response = self.outpaint(origin_task_id, zoom_ratio, prompt)
        if outpaint_response.status_code == 200:
            task_id = outpaint_response.json().get('task_id')
            retries = 0
            while retries < max_retries:
                fetch_response = self.fetch(task_id)
                if fetch_response.status_code == 200:
                    image_url = fetch_response.json().get('task_result', {}).get('image_url')
                    if image_url:
                        return task_id, image_url
                    else:
                        time.sleep(30)
                        retries += 1
                else:
                    raise Exception(f"Fetch failed with status {fetch_response.status_code}: {fetch_response.json()}")
            raise Exception("Max retries reached without getting an image URL.")
        else:
            raise Exception(f"Outpaint failed with status {outpaint_response.status_code}: {outpaint_response.json()}")
    
