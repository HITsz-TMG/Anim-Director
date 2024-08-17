from imgurpython import ImgurClient
import os
import time
import requests

class Imgur:
    def __init__(self, client_id = '', client_secret = '', access_token = '', refresh_token = ''):
        self.client = ImgurClient(client_id, client_secret, access_token, refresh_token)
    

    def upload_image(self, image_path, album_id):
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)

        config = {
            'album': album_id,
            'name': name,
            'title': name,
        }
        
        attempt = 0
        max_attempts = 10  # Maximum number of upload attempts
        while attempt < max_attempts:
            try:
                print("Uploading image to album...")
                image_up = self.client.upload_from_path(image_path, config=config, anon=False)
                print("Image uploaded.")
                print(f"Id of image: {image_up['id']}")
                print(f"Link to image: {image_up['link']}")
                time.sleep(30)
                return image_up
            except Exception as e:
                print(f"Upload failed: {e}")
                attempt += 1
                if attempt < max_attempts:
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print("Maximum upload attempts reached. Upload failed.")
                    break

                
    def download_image(self, image_id, save_directory):
        print("Downloading image from album...")
        image_info = self.client.get_image(image_id)
        image_url = image_info.link
        response = requests.get(image_url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})

        if response.status_code == 200:
            file_path = os.path.join(save_directory, f"{image_info.id}.png")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Image saved as {file_path}")
            return file_path
        else:
            print("Failed to download image")
            return None
