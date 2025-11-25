import pandas as pd
import os
import shutil
import requests as req
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import time
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from notifications import send_notification
from image_utils import get_file_size_in_mb, resize_with_aspect_ratio
from PIL import UnidentifiedImageError

# get JOB_ID from environment
CWD = os.getcwd()

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{CWD}/link_check.log', level=logging.INFO, filemode='w')

GBIF_MULTIMEDIA_DATA = "/projectnb/herbdl/data/harvard-herbaria/gbif/multimedia.txt"

num_invalid_links = 0
invalid_links = []
ids = []

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
]

session = req.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

min_delay = 10
max_delay = 30

def process_row(row):
    global num_invalid_links, invalid_links, ids
    gbif_id = row['gbifID']
    image_url = row['identifier']

    try:
        image_response = session.get(image_url, stream=True, timeout=60, headers={
            "User-Agent": random.choice(user_agents),
            "Connection": "keep-alive",
            "Referer": "https://scc-ondemand1.bu.edu/"
        })

        if image_response.status_code == 200:

            if image_response.headers.get('Content-Type') != 'image/jpeg':
                logger.error(f"Invalid content type for {gbif_id}: {image_response.headers.get('Content-Type')}. Skipping. ")

                ids.append(gbif_id)
                num_invalid_links += 1
                invalid_links.append(image_url)
            
                del image_response
            
    except Exception as e:
        logger.error(f"Error downloading {gbif_id}: {e}")

if __name__ == "__main__":

    df = pd.read_csv(GBIF_MULTIMEDIA_DATA, delimiter="\t", usecols=['gbifID', 'identifier'], on_bad_lines='skip')

    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(process_row, row) for index, row in df.iterrows()]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Generated an exception: {exc}")