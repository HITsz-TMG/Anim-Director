"""MiniMax Hailuo image-to-video API tool.

This module provides a first-party image-to-video backend backed by the
MiniMax video generation API. It implements the full asynchronous workflow:
image-to-video task creation from a first frame image, task-status polling and
generated video file retrieval. Both the global and the Mainland China API
hosts are supported through the ``region`` argument.
"""

import os
import time
import requests
import urllib.request

# Set your MiniMax API key here, or provide it via the ``MINIMAX_API_KEY``
# environment variable.
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "your_minimax_key_here")

# API hosts for the supported regions.
REGION_BASE_URLS = {
    "global_en": "https://api.minimax.io",
    "cn_zh": "https://api.minimaxi.com",
}

DEFAULT_MODEL = "MiniMax-Hailuo-2.3"
SUPPORTED_MODELS = [
    "MiniMax-Hailuo-2.3",
    "MiniMax-Hailuo-2.3-Fast",
    "MiniMax-Hailuo-02",
    "T2V-01-Director",
    "T2V-01",
    "I2V-01-Director",
    "I2V-01-live",
    "I2V-01",
]


def _base_url(region):
    if region not in REGION_BASE_URLS:
        raise ValueError(
            f"Unknown region '{region}'. Expected one of: {list(REGION_BASE_URLS)}"
        )
    return REGION_BASE_URLS[region]


def _headers(api_key):
    return {
        "Authorization": f"Bearer {api_key or MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }


def image_to_video(first_frame_image, prompt=None, model=DEFAULT_MODEL,
                   duration=None, resolution=None, prompt_optimizer=None,
                   fast_pretreatment=None, callback_url=None,
                   region="global_en", api_key=None, download_dir=None,
                   max_retries=60, poll_interval=5):
    """Create an image-to-video task from a first frame image.

    ``first_frame_image`` may be a publicly reachable URL or a
    ``data:image/...;base64,...`` string, following the MiniMax API contract.
    When ``download_dir`` is provided the task is polled and the resulting
    video is downloaded once the generation succeeds.
    """
    url = f"{_base_url(region)}/v1/video_generation"
    data = {
        "model": model,
        "first_frame_image": first_frame_image,
    }
    if prompt is not None:
        data["prompt"] = prompt
    if prompt_optimizer is not None:
        data["prompt_optimizer"] = prompt_optimizer
    if fast_pretreatment is not None:
        data["fast_pretreatment"] = fast_pretreatment
    if duration is not None:
        data["duration"] = duration
    if resolution is not None:
        data["resolution"] = resolution
    if callback_url is not None:
        data["callback_url"] = callback_url

    print("Request URL:", url)
    print("Data:", data)

    response = requests.post(url, headers=_headers(api_key), json=data)

    if response.status_code != 200:
        print("Error: API request failed with status code:", response.status_code)
        print("Response Text:", response.text)
        return {"error": "API request failed", "status_code": response.status_code, "text": response.text}

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Could not decode JSON response.")
        print("Response Text:", response.text)
        return {"error": "Could not decode JSON", "text": response.text}

    status_code = result.get("base_resp", {}).get("status_code")
    if status_code not in (0, None):
        print("Error: API returned status_code:", status_code)
        print("Response:", result)
        return result

    task_id = result.get("task_id")
    if task_id:
        print(f"Task created with task_id={task_id}. Polling for completion...")
        get_generation(task_id, region=region, api_key=api_key,
                       download_dir=download_dir, max_retries=max_retries,
                       poll_interval=poll_interval)
    return result


def query_video_generation(task_id, region="global_en", api_key=None):
    """Query the status of a video generation task by ``task_id``."""
    url = f"{_base_url(region)}/v1/query/video_generation"
    response = requests.get(url, headers=_headers(api_key), params={"task_id": task_id})

    if response.status_code != 200:
        print("Error: API request failed with status code:", response.status_code)
        print("Response Text:", response.text)
        return {"error": "API request failed", "status_code": response.status_code, "text": response.text}

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Could not decode JSON response.")
        print("Response Text:", response.text)
        return {"error": "Could not decode JSON", "text": response.text}


def retrieve_video(file_id, region="global_en", api_key=None):
    """Retrieve the download information for a generated video ``file_id``."""
    url = f"{_base_url(region)}/v1/files/retrieve"
    response = requests.get(url, headers=_headers(api_key), params={"file_id": file_id})

    if response.status_code != 200:
        print("Error: API request failed with status code:", response.status_code)
        print("Response Text:", response.text)
        return {"error": "API request failed", "status_code": response.status_code, "text": response.text}

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Could not decode JSON response.")
        print("Response Text:", response.text)
        return {"error": "Could not decode JSON", "text": response.text}


def get_generation(task_id, region="global_en", api_key=None, download_dir=None,
                   max_retries=60, poll_interval=5):
    """Poll a task until it finishes and optionally download the video.

    Returns the final query payload. When the task succeeds and
    ``download_dir`` is set, the generated ``file_id`` is resolved to a
    download URL through the file-retrieve endpoint and saved to disk.
    """
    retries = 0
    while retries <= max_retries:
        data = query_video_generation(task_id, region=region, api_key=api_key)
        if "error" in data:
            return data

        status = data.get("status")
        if status in ("Queueing", "Preparing", "Processing"):
            if retries < max_retries:
                print(f"Generation in '{status}' state. Retrying in {poll_interval} seconds...")
                time.sleep(poll_interval)
                retries += 1
                continue
            raise Exception(f"Maximum retries reached while waiting for generation. Last status: {status}")
        if status == "Fail":
            raise Exception(f"Generation failed: {data}")

        # Success: resolve the file and optionally download it.
        file_id = data.get("file_id")
        if download_dir is None:
            download_dir = os.getcwd()
        if file_id and download_dir:
            file_info = retrieve_video(file_id, region=region, api_key=api_key)
            download_url = file_info.get("file", {}).get("download_url")
            if download_url:
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                    print(f"Created directory: {download_dir}")
                download_path = os.path.join(download_dir, f"{task_id}.mp4")
                print(f"Downloading video from {download_url} to {download_path}...")
                try:
                    urllib.request.urlretrieve(download_url, download_path)
                    print(f"Successfully downloaded video to {download_path}")
                except Exception as e:
                    print(f"Failed to download video: {str(e)}")
            data["download_info"] = file_info
        return data

    return {"error": "Maximum retries reached"}


# Example usage:
# response = image_to_video(
#     "https://example.com/first_frame.png",
#     "A fox and a boy dancing together.",
#     region="global_en",
#     download_dir="./outputs",
# )
# print("Response:", response)
