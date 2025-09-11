import os
import time
import requests
import urllib.request

VIDU_API_KEY = "your_vidu_key_here"


def image_to_video(image_paths, prompt, model="vidu2.0", duration=4, seed=0, resolution="720p", movement_amplitude="auto", download_dir=None):
    """
    Generates a video from a single image with more parameters.
    """
    url = "https://api.vidu.com/ent/v2/img2video"
    headers = {
        "Authorization": f"Token {VIDU_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "images": image_paths,  # 确保 URL 没有多余的空格
        "prompt": prompt,
        "duration": duration,
        "seed": seed,
        "resolution": resolution,
        "movement_amplitude": movement_amplitude
    }
    print("Request URL:", url)
    print("Headers:", headers)
    print("Data:", data)

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        try:
            result = response.json()
            if 'task_id' in result:
                # Wait 30 seconds before checking the generation
                print(f"Waiting 30 seconds before checking generation...")
                time.sleep(30)
                get_generation(result['task_id'], download_dir=download_dir)
            return result
        except requests.exceptions.JSONDecodeError:
            print("Error: Could not decode JSON response.")
            print("Response Text:", response.text)
            return {"error": "Could not decode JSON", "text": response.text}
    else:
        print("Error: API request failed with status code:", response.status_code)
        print("Response Text:", response.text)
        return {"error": "API request failed", "status_code": response.status_code, "text": response.text}
    

def reference_to_video(image_paths, prompt, model="vidu2.0", duration=4, seed=0, resolution="720p", movement_amplitude="auto", download_dir=None):
    """
    Generates a video from a reference video and a reference image.
    """
    url = "https://api.vidu.com/ent/v2/reference2video"
    headers = {
        "Authorization": f"Token {VIDU_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "images": image_paths,
        "prompt": prompt,
        "duration": duration,
        "seed": seed,
        "resolution": resolution,
        "movement_amplitude": movement_amplitude
    }
    print("Request URL:", url)
    print("Headers:", headers)
    print("Data:", data)

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        try:
            result = response.json()
            if 'task_id' in result:
                # Wait 30 seconds before checking the generation
                print(f"Waiting 30 seconds before checking generation...")
                time.sleep(30)
                get_generation(result['task_id'], download_dir=download_dir)
            return result
        except requests.exceptions.JSONDecodeError:
            print("Error: Could not decode JSON response.")
            print("Response Text:", response.text)
            return {"error": "Could not decode JSON", "text": response.text}
    else:
        print("Error: API request failed with status code:", response.status_code)
        print("Response Text:", response.text)
        return {"error": "API request failed", "status_code": response.status_code, "text": response.text}


def start_end_to_video(image_paths, prompt, model="vidu2.0", duration=4, seed=0, resolution="720p", movement_amplitude="auto", download_dir=None):
    """
    Generates a video from a start and end frame.
    """
    url = "https://api.vidu.com/ent/v2/start-end2video"
    headers = {
        "Authorization": f"Token {VIDU_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "images": image_paths,
        "prompt": prompt,
        "duration": duration,
        "seed": seed,
        "resolution": resolution,
        "movement_amplitude": movement_amplitude
    }
    print("Request URL:", url)
    print("Headers:", headers)
    print("Data:", data)

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        try:
            result = response.json()
            if 'task_id' in result:
                # Wait 30 seconds before checking the generation
                print(f"Waiting 30 seconds before checking generation...")
                time.sleep(30)
                get_generation(result['task_id'], download_dir=download_dir)
            return result
        except requests.exceptions.JSONDecodeError:
            print("Error: Could not decode JSON response.")
            print("Response Text:", response.text)
            return {"error": "Could not decode JSON", "text": response.text}
    else:
        print("Error: API request failed with status code:", response.status_code)
        print("Response Text:", response.text)
        return {"error": "API request failed", "status_code": response.status_code, "text": response.text}


def upscale(generation_id, model="vidu1.0", download_dir=None):
    """
    Upscales a video.
    """
    # First get the creation_id from the generation
    generation_data = get_generation(generation_id)
    
    if "creations" not in generation_data or not generation_data["creations"]:
        return {"error": "No creations found for this generation"}
    
    # Get the first creation_id
    creation_id = generation_data["creations"][0]['id']
    
    url = f"https://api.vidu.com/ent/v2/upscale"
    headers = {
        "Authorization": f"Token {VIDU_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "creation_id": creation_id,  # Fixed typo from 'creantion_id'
        "model": model
    }
    print("Request URL:", url)
    print("Headers:", headers)
    print("Data:", data)

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        try:
            result = response.json()
            if 'task_id' in result:
                # Wait 30 seconds before checking the generation
                print(f"Waiting 30 seconds before checking generation...")
                time.sleep(30)
                get_generation(result['task_id'], download_dir=download_dir)
            return result
        except requests.exceptions.JSONDecodeError:
            print("Error: Could not decode JSON response.")
            print("Response Text:", response.text)
            return {"error": "Could not decode JSON", "text": response.text}
    else:
        print("Error: API request failed with status code:", response.status_code)
        print("Response Text:", response.text)
        return {"error": "API request failed", "status_code": response.status_code, "text": response.text}
    

def get_generation(generation_id, max_retries=60, download_dir=None):
    """
    Gets a generation and optionally downloads the video if download_dir is specified.
    """
    url = f"https://api.vidu.com/ent/v2/tasks/{generation_id}/creations"
    headers = {
        "Authorization": f"Token {VIDU_API_KEY}",
        "Content-Type": "application/json"
    }
    
    retries = 0
    while retries <= max_retries:
        # print("Request URL:", url)
        # print("Headers:", headers)
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if "state" in data:
                    state = data["state"]
                    if state == "failed":
                        raise Exception(f"Generation failed: {data.get('message', 'No error message provided')}")
                    elif state in ["created", "queueing", "processing"]:
                        if retries < max_retries:
                            print(f"Generation in '{state}' state. Retrying in 5 seconds...")
                            time.sleep(5)
                            retries += 1
                            continue
                        else:
                            raise Exception(f"Maximum retries reached while waiting for generation. Last state: {state}")
                
                # If download_dir is provided and we have creations with URLs, download the videos
                if download_dir is None:
                    download_dir = os.getcwd()
                if download_dir and "creations" in data and data["creations"]:
                    if not os.path.exists(download_dir):
                        os.makedirs(download_dir)
                        print(f"Created directory: {download_dir}")
                        
                    for idx, creation in enumerate(data["creations"]):
                        if "url" in creation and creation["url"]:
                            video_url = creation["url"]
                            file_name = f"{generation_id}_{idx}.mp4"
                            download_path = os.path.join(download_dir, file_name)
                            
                            print(f"Downloading video from {video_url} to {download_path}...")
                            try:
                                urllib.request.urlretrieve(video_url, download_path)
                                print(f"Successfully downloaded video to {download_path}")
                            except Exception as e:
                                print(f"Failed to download video: {str(e)}")
                
                return data
            except requests.exceptions.JSONDecodeError:
                print("Error: Could not decode JSON response.")
                print("Response Text:", response.text)
                return {"error": "Could not decode JSON", "text": response.text}
        else:
            print("Error: API request failed with status code:", response.status_code)
            print("Response Text:", response.text)
            return {"error": "API request failed", "status_code": response.status_code, "text": response.text}
    
    return {"error": "Maximum retries reached"}

    
# response = reference_to_video(["https://i.imgur.com/TJHiAqU.png", "https://i.imgur.com/JCXToSh.png"], "A fox and a boy dancing together.")
# print("Response:", response)

# response = upscale("795985659562041344")
# print("Response:", response)