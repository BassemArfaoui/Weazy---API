import os
import uuid
import requests
import random
from app.core.config import  DOWNLOADS_DIR
from app.core.config import SUCCESS_MESSAGES, ERROR_MESSAGES


def download_image(url):
    temp_filename = os.path.join(DOWNLOADS_DIR, f"{uuid.uuid4().hex}.jpg")
    response = requests.get(url)
    if response.status_code == 200:
        with open(temp_filename, "wb") as f:
            f.write(response.content)
        return temp_filename
    else:
        raise Exception("Failed to download image")



def get_random_message(success: bool) -> str:
    return random.choice(SUCCESS_MESSAGES if success else ERROR_MESSAGES)



