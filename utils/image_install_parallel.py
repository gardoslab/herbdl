import pandas as pd
import os
import shutil
import requests as req
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from notifications import send_notification
from image_utils import get_file_size_in_mb, resize_with_aspect_ratio
from PIL import UnidentifiedImageError

import datetime as dt

"""
Image install script to download images from a GBIF multimedia.txt file. 
Accurate as of September Fall 2025.
"""

CWD = os.getcwd()

today = dt.datetime.now().strftime("%Y-%m-%d")

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{CWD}/image_install_{today}.log', level=logging.INFO, filemode='w')

link_logger = logging.getLogger("link_logger")
link_logger.setLevel(logging.INFO)

INSTALL_PATH = "/projectnb/herbdl/data/GBIF-F25/images"
GBIF_MULTIMEDIA_DATA = "/projectnb/herbdl/data/GBIF-F25/multimedia.txt"

existing_gbif_datasets = ["/projectnb/herbdl/data/harvard-herbaria/gbif/multimedia.txt", "/projectnb/herbdl/data/GBIF-F24/multimedia.txt"]
existing_gbif_dfs = [pd.read_csv(f, delimiter="\t", usecols=['gbifID']) for f in existing_gbif_datasets]

existing_gbif_ids = set()

for df in existing_gbif_dfs:
    existing_gbif_ids.update(df['gbifID'].astype(str).tolist())

print(list(existing_gbif_ids)[:10])

print(f"Number of existing ids to check for duplicates: {len(existing_gbif_ids)}")

n_installed = len(os.listdir(INSTALL_PATH))
print(f"Number of already installed images: {n_installed}")

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

min_delay = 15
max_delay = 30

def is_duplicate(gbif_id):
    return str(gbif_id) in existing_gbif_ids
    
def download_image(gbif_id, image_url, local_path):

    try:
        image_response = session.get(image_url, stream=True, verify=False, headers={
            "User-Agent": random.choice(user_agents),
            "Connection": "keep-alive",
            "Referer": "https://scc-ondemand1.bu.edu/"
        })

        if image_response.status_code == 200:

            if image_response.headers.get('Content-Type') not in ['image/jpeg', 'image/jpg', "image/tiff", "image/png"] or "image" not in image_response.headers.get('Content-Type', ""):
                logger.error(f"Invalid content type for {gbif_id} from {image_url}: {image_response.headers.get('Content-Type')}. Skipping. ")
                del image_response
                return False

            with open(local_path, 'wb') as out_file:
                shutil.copyfileobj(image_response.raw, out_file)

            logger.info(f"Downloaded {gbif_id} to {local_path}")
            del image_response

            return True
        else:
            raise Exception(f"HTTP {image_response.status_code}")

    except Exception as e:
        logger.error(f"Error downloading {gbif_id} from {image_url}: {e}")
        return False


def resize_image(gbif_id, local_path):
    result = resize_with_aspect_ratio(local_path, local_path)
    if result:
        logger.info(f"Resized {gbif_id} to 1000x1000. Path: {local_path}")
    else:
        logger.info(f"Skipped resizing {gbif_id} as it is already 1000x1000")
    

def process_row(row):
    global n_installed
    gbif_id = row['gbifID']
    image_url = row['identifier']
    local_path = os.path.join(INSTALL_PATH, f"{gbif_id}.jpg")

    downloaded=False

    if is_duplicate(gbif_id):
        logger.warning(f"Image {gbif_id} is a duplicate, skipping download.")
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.warning(f"Removed existing file for duplicate {gbif_id} at {local_path}.")
        return

    if os.path.exists(local_path):
        logger.warning(f"Image {gbif_id} already exists in {local_path}, checking size...")
        size = get_file_size_in_mb(local_path)
        if size < 0.01:
            logger.warning(f"Image {gbif_id} is too small ({size} MB), redownloading")
            downloaded = download_image(gbif_id, image_url, local_path)
    else:
        logger.info(f"Downloading {gbif_id} to {local_path}. Image URL: {image_url}")
        downloaded = download_image(gbif_id, image_url, local_path)

    if downloaded:
        n_installed += 1

    if n_installed % 50000 == 0 and n_installed > 0:
        send_notification("Image Installation", f"Installed {n_installed} images. Remaining: {len(df) - n_installed}")
        logger.info(f"Installed {n_installed} images")

    try:
        if downloaded:
            resize_image(gbif_id, local_path)

    except (OSError, UnidentifiedImageError) as e:            
        os.remove(local_path)
        logger.error(f"Error resizing {gbif_id}: {e}.")
        # process_row(row)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--country", dest="country",
                        help="Country to download samples from", metavar="COUNTRY CODE")

    args = parser.parse_args()
    country = args.country

    df = pd.read_csv(GBIF_MULTIMEDIA_DATA, delimiter="\t", usecols=['gbifID', 'identifier'], on_bad_lines='skip')
    print(f"Length of multimedia.txt: {len(df)}")

    if country:
        df = df[df['countryCode'] == country]

    send_notification("Image Installation", f"Starting image installation for {len(df)} images")

    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(process_row, row) for index, row in df.iterrows()]

        for future in as_completed(futures):
            try:
                future.result()

            except KeyboardInterrupt:
                logger.error("Process interrupted by user. Installed {n_installed} images so far. Exiting...")
                print(f"Installed {n_installed} images")
                executor.shutdown(wait=False)
                
            except Exception as exc:
                logger.error(f"Generated an exception: {exc}")

    print(f'All done. Number of installed images: {n_installed}')