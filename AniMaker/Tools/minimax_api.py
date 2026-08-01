import os
import time
import requests
import urllib.request


MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "your_minimax_key_here")

REGION_BASE_URLS = {
    "global_en": "https://api.minimax.io",
    "cn_zh": "https://api.minimaxi.com",
}

DEFAULT_MODEL = "MiniMax-H3"
SUPPORTED_MODELS = [DEFAULT_MODEL]
SUPPORTED_RATIOS = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
SUPPORTED_DURATIONS = list(range(4, 16))
SUPPORTED_CONTENT_TYPES = ["text", "image_url", "video_url", "audio_url"]
SUPPORTED_CONTENT_ROLES = [
    "first_frame",
    "last_frame",
    "reference_image",
    "reference_video",
    "reference_audio",
]


def _base_url(region):
    if region not in REGION_BASE_URLS:
        raise ValueError(f"Unknown region '{region}'. Expected one of: {list(REGION_BASE_URLS)}")
    return REGION_BASE_URLS[region]


def _headers(api_key):
    return {
        "Authorization": f"Bearer {api_key or MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }


def _clean_payload(data):
    return {key: value for key, value in data.items() if value is not None}


def _content_roles(content):
    return [item.get("role") for item in content if isinstance(item, dict) and item.get("role")]


def _validate_content(content):
    if not content or not any(item.get("type") == "text" and item.get("text") for item in content if isinstance(item, dict)):
        raise ValueError("content must include one non-empty text item")

    roles = set(_content_roles(content))
    frame_roles = {"first_frame", "last_frame"}
    reference_roles = {"reference_image", "reference_video", "reference_audio"}

    if roles & frame_roles and roles & reference_roles:
        raise ValueError("image-to-video and reference-to-video content roles cannot be mixed")

    if "reference_audio" in roles and not (roles & {"reference_image", "reference_video"}):
        raise ValueError("reference_audio requires at least one reference image or reference video")


class MiniMaxVideoAPI:
    def __init__(self, api_key=None, region="global_en", model=DEFAULT_MODEL):
        self.api_key = api_key or MINIMAX_API_KEY
        self.region = region
        self.model = model

    def _request(self, method, path, region=None, params=None, json_body=None):
        url = f"{_base_url(region or self.region)}{path}"
        response = requests.request(
            method=method,
            url=url,
            headers=_headers(self.api_key),
            params=params,
            json=json_body,
        )
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            return {
                "error": "Could not decode JSON",
                "status_code": response.status_code,
                "text": response.text,
            }

        if response.status_code != 200:
            return {
                "error": "API request failed",
                "status_code": response.status_code,
                "data": data,
            }
        return data

    @staticmethod
    def text_content(text):
        return {"type": "text", "text": text}

    @staticmethod
    def image_content(image_url, role=None):
        content = {"type": "image_url", "image_url": {"url": image_url}}
        if role:
            content["role"] = role
        return content

    @staticmethod
    def video_content(video_url, role=None):
        content = {"type": "video_url", "video_url": {"url": video_url}}
        if role:
            content["role"] = role
        return content

    @staticmethod
    def audio_content(audio_url, role=None):
        content = {"type": "audio_url", "audio_url": {"url": audio_url}}
        if role:
            content["role"] = role
        return content

    def create_video_generation(
        self,
        content,
        resolution="2K",
        duration=4,
        ratio="adaptive",
        callback_url=None,
        model=None,
        region=None,
        **regional_fields,
    ):
        _validate_content(content)
        payload = _clean_payload(
            {
                "model": model or self.model,
                "content": content,
                "resolution": resolution,
                "duration": duration,
                "ratio": ratio,
                "callback_url": callback_url,
                **regional_fields,
            }
        )
        return self._request("POST", "/v2/video_generation", region=region, json_body=payload)

    def text_to_video(
        self,
        prompt,
        resolution="2K",
        duration=4,
        ratio="16:9",
        callback_url=None,
        model=None,
        region=None,
        **regional_fields,
    ):
        return self.create_video_generation(
            content=[self.text_content(prompt)],
            resolution=resolution,
            duration=duration,
            ratio=ratio,
            callback_url=callback_url,
            model=model,
            region=region,
            **regional_fields,
        )

    def image_to_video(
        self,
        prompt,
        image_url,
        role="first_frame",
        resolution="2K",
        duration=4,
        callback_url=None,
        model=None,
        region=None,
        **regional_fields,
    ):
        content = [self.text_content(prompt), self.image_content(image_url, role=role)]
        return self.create_video_generation(
            content=content,
            resolution=resolution,
            duration=duration,
            ratio="adaptive",
            callback_url=callback_url,
            model=model,
            region=region,
            **regional_fields,
        )

    def reference_to_video(
        self,
        prompt,
        reference_image_urls=None,
        reference_video_urls=None,
        reference_audio_urls=None,
        resolution="2K",
        duration=4,
        ratio="adaptive",
        callback_url=None,
        model=None,
        region=None,
        **regional_fields,
    ):
        content = [self.text_content(prompt)]
        for image_url in reference_image_urls or []:
            content.append(self.image_content(image_url, role="reference_image"))
        for video_url in reference_video_urls or []:
            content.append(self.video_content(video_url, role="reference_video"))
        for audio_url in reference_audio_urls or []:
            content.append(self.audio_content(audio_url, role="reference_audio"))
        return self.create_video_generation(
            content=content,
            resolution=resolution,
            duration=duration,
            ratio=ratio,
            callback_url=callback_url,
            model=model,
            region=region,
            **regional_fields,
        )

    def query_video_generation(self, task_id, region=None):
        return self._request(
            "GET",
            f"/v2/query/video_generation/{task_id}",
            region=region,
        )

    def list_video_generation(
        self,
        page_num=1,
        page_size=20,
        filter_status=None,
        filter_task_ids=None,
        filter_model=None,
        filter_task_type=None,
        region=None,
    ):
        params = _clean_payload(
            {
                "page_num": page_num,
                "page_size": page_size,
                "filter.status": filter_status,
                "filter.task_ids": filter_task_ids,
                "filter.model": filter_model,
                "filter.task_type": filter_task_type,
            }
        )
        return self._request("GET", "/v2/query/video_generation", region=region, params=params)

    def delete_video_generation(self, task_id, region=None):
        return self._request(
            "DELETE",
            f"/v2/video_generation/{task_id}",
            region=region,
        )

    def get_generation(self, task_id, region=None, download_dir=None, max_retries=60, poll_interval=5):
        retries = 0
        while retries <= max_retries:
            data = self.query_video_generation(task_id, region=region)
            if "error" in data:
                return data

            task = data.get("task", {})
            status = task.get("status")
            if status in {"queued", "running"}:
                if retries >= max_retries:
                    raise Exception(f"Maximum retries reached while waiting for generation. Last status: {status}")
                time.sleep(poll_interval)
                retries += 1
                continue

            if status in {"failed", "expired", "cancelled"}:
                return data

            if download_dir:
                video_url = task.get("content", {}).get("url")
                if video_url:
                    if not os.path.exists(download_dir):
                        os.makedirs(download_dir)
                    download_path = os.path.join(download_dir, f"{task_id}.mp4")
                    urllib.request.urlretrieve(video_url, download_path)
                    data["download_path"] = download_path
            return data

        return {"error": "Maximum retries reached"}


_DEFAULT_CLIENT = MiniMaxVideoAPI()


def text_content(text):
    return MiniMaxVideoAPI.text_content(text)


def image_content(image_url, role=None):
    return MiniMaxVideoAPI.image_content(image_url, role=role)


def video_content(video_url, role=None):
    return MiniMaxVideoAPI.video_content(video_url, role=role)


def audio_content(audio_url, role=None):
    return MiniMaxVideoAPI.audio_content(audio_url, role=role)


def create_video_generation(content, resolution="2K", duration=4, ratio="adaptive", callback_url=None, model=DEFAULT_MODEL, region="global_en", **regional_fields):
    client = MiniMaxVideoAPI(region=region, model=model)
    return client.create_video_generation(
        content=content,
        resolution=resolution,
        duration=duration,
        ratio=ratio,
        callback_url=callback_url,
        model=model,
        region=region,
        **regional_fields,
    )


def text_to_video(prompt, resolution="2K", duration=4, ratio="16:9", callback_url=None, model=DEFAULT_MODEL, region="global_en", **regional_fields):
    client = MiniMaxVideoAPI(region=region, model=model)
    return client.text_to_video(
        prompt=prompt,
        resolution=resolution,
        duration=duration,
        ratio=ratio,
        callback_url=callback_url,
        model=model,
        region=region,
        **regional_fields,
    )


def image_to_video(prompt, image_url, role="first_frame", resolution="2K", duration=4, callback_url=None, model=DEFAULT_MODEL, region="global_en", **regional_fields):
    client = MiniMaxVideoAPI(region=region, model=model)
    return client.image_to_video(
        prompt=prompt,
        image_url=image_url,
        role=role,
        resolution=resolution,
        duration=duration,
        callback_url=callback_url,
        model=model,
        region=region,
        **regional_fields,
    )


def reference_to_video(
    prompt,
    reference_image_urls=None,
    reference_video_urls=None,
    reference_audio_urls=None,
    resolution="2K",
    duration=4,
    ratio="adaptive",
    callback_url=None,
    model=DEFAULT_MODEL,
    region="global_en",
    **regional_fields,
):
    client = MiniMaxVideoAPI(region=region, model=model)
    return client.reference_to_video(
        prompt=prompt,
        reference_image_urls=reference_image_urls,
        reference_video_urls=reference_video_urls,
        reference_audio_urls=reference_audio_urls,
        resolution=resolution,
        duration=duration,
        ratio=ratio,
        callback_url=callback_url,
        model=model,
        region=region,
        **regional_fields,
    )


def query_video_generation(task_id, region="global_en"):
    return _DEFAULT_CLIENT.query_video_generation(task_id, region=region)


def list_video_generation(page_num=1, page_size=20, filter_status=None, filter_task_ids=None, filter_model=None, filter_task_type=None, region="global_en"):
    return _DEFAULT_CLIENT.list_video_generation(
        page_num=page_num,
        page_size=page_size,
        filter_status=filter_status,
        filter_task_ids=filter_task_ids,
        filter_model=filter_model,
        filter_task_type=filter_task_type,
        region=region,
    )


def delete_video_generation(task_id, region="global_en"):
    return _DEFAULT_CLIENT.delete_video_generation(task_id, region=region)
