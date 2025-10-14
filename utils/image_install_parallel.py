import pandas as pd
import os
import shutil
import requests as req
from requests.exceptions import ConnectTimeout, ReadTimeout, Timeout
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
from urllib.parse import urlparse
import threading
import time

HOST_COOLDOWN_DEFAULT = 30 * 60
HOST_COOLDOWN_TIMEOUT = 60 * 60
host_block_until = {}
host_lock = threading.Lock()
counter_lock = threading.Lock()


"""
Image install script to download images from a GBIF multimedia.txt file. 
Accurate as of September Fall 2025.
"""

CWD = os.getcwd()
LOG_DIR = "/projectnb/herbdl/logs"

CHECKPOINT_FILE = os.path.join(CWD, "processed_ids.txt")
FAILED_FILE = os.path.join(CWD, "failed_ids.txt")

checkpoint_lock = threading.Lock()

if os.path.exists(FAILED_FILE):
    with open(FAILED_FILE) as f:
        failed_ids = {line.strip() for line in f if line.strip()}
else:
    failed_ids = set()

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        processed_ids = {line.strip() for line in f if line.strip()}
else:
    processed_ids = set()


today = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=f'{LOG_DIR}/image_install_{today}.log',
                    level=logging.INFO,
                    filemode='w')
logger = logging.getLogger(__name__)


link_logger = logging.getLogger("link_logger")
link_logger.setLevel(logging.INFO)

INSTALL_PATH = "/projectnb/herbdl/data/GBIF-F25h"
GBIF_MULTIMEDIA_DATA = "/projectnb/herbdl/data/GBIF-F25/multimedia.txt"


existing_gbif_datasets = ["/projectnb/herbdl/data/harvard-herbaria/gbif/multimedia.txt", "/projectnb/herbdl/data/GBIF-F24/multimedia.txt"]
existing_gbif_dfs = [pd.read_csv(f, delimiter="\t", usecols=['gbifID']) for f in existing_gbif_datasets]

existing_gbif_ids = set()

for df in existing_gbif_dfs:
    existing_gbif_ids.update(df['gbifID'].astype(str).tolist())

print(list(existing_gbif_ids)[:10])

print(f"Number of existing ids to check for duplicates: {len(existing_gbif_ids)}")

n_installed = sum(len(files) for _, _, files in os.walk(INSTALL_PATH))
print(f"Number of already installed images: {n_installed}")



user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
]

session = req.Session()
retry_strategy = Retry(
    total=2,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

min_delay = 15
max_delay = 30

def get_hierarchical_path(base_dir, gbif_id, ext=".jpg"):
    stem = str(gbif_id)
    prefix1 = stem[:3] if len(stem) >= 3 else stem
    prefix2 = stem[3:6] if len(stem) >= 6 else "000"

    dest_dir = os.path.join(base_dir, prefix1, prefix2)
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, f"{stem}{ext}")


def _host_from_url(url):
    return urlparse(url).netloc.split(":")[0]

def is_host_blocked(url):
    host = _host_from_url(url)
    now = time.time()
    with host_lock:
        until = host_block_until.get(host)
        if until and now < until:
            return True
        if until and now >= until:
            del host_block_until[host]
    return False

def block_host(url, retry_after=None, timeout_issue=False):
    host = _host_from_url(url)
    now = time.time()
    seconds = HOST_COOLDOWN_TIMEOUT if timeout_issue else HOST_COOLDOWN_DEFAULT
    if retry_after and not timeout_issue:
        try:
            seconds = int(retry_after)
        except Exception:
            try:
                from email.utils import parsedate_to_datetime
                dt_retry = parsedate_to_datetime(retry_after)
                seconds = max(0, (dt_retry - dt.datetime.now(dt.timezone.utc)).total_seconds())

            except Exception:
                seconds = HOST_COOLDOWN_DEFAULT
    with host_lock:
        host_block_until[host] = now + seconds
    reason = "timeout issues" if timeout_issue else "rate limiting"
    logger.warning(f"Blocking host '{host}' due to {reason}. Cooling down for ~{int(seconds)}s.")


def is_duplicate(gbif_id):
    return str(gbif_id) in existing_gbif_ids
    
def download_image_from_candidates(gbif_id, candidate_urls, local_path):
    """
    Try each URL for this gbif_id until one succeeds.
    Skips hosts under cooldown; on 429, cools down that host and tries the next.
    """
    random.shuffle(candidate_urls)
    for image_url in candidate_urls:
        if is_host_blocked(image_url):
            logger.info(f"Host on cooldown; skipping for {gbif_id}: {image_url}")
            continue

        try:
            time.sleep(random.uniform(0.2, 0.8))
            image_response = session.get(
                image_url,
                stream=True,
                verify=False,
                headers={
                    "User-Agent": random.choice(user_agents),
                    "Connection": "keep-alive",
                    "Referer": "https://scc-ondemand1.bu.edu/",
                },
                timeout=60,
            )

            status = image_response.status_code

            if status == 429:
                block_host(image_url, image_response.headers.get("Retry-After"))
                del image_response
                continue

            if status >= 500:
                logger.error(f"Server error {status} for {gbif_id} from {image_url}; trying another source.")
                del image_response
                continue

            if status != 200:
                logger.error(f"HTTP {status} for {gbif_id} from {image_url}; trying another source.")
                del image_response
                continue

            ctype = (image_response.headers.get("Content-Type") or "").lower()
            if not ctype:
                logger.warning(f"Missing Content-Type header for {gbif_id} from {image_url}, attempting download anyway.")
            elif ctype and any(bad in ctype for bad in ["text/html", "text/plain", "application/json", "application/xml"]):
                logger.error(f"Invalid content type for {gbif_id} from {image_url}: {ctype}. Skipping.")
                del image_response
                continue

            with open(local_path, "wb") as out_file:
                shutil.copyfileobj(image_response.raw, out_file)

            logger.info(f"Downloaded {gbif_id} to {local_path} from {image_url}")
            del image_response
            return True

        except (ConnectTimeout, ReadTimeout, Timeout) as e:
            logger.error(f"Timeout error for {gbif_id} from {image_url}: {e}")
            block_host(image_url, timeout_issue=True)
            continue
        except Exception as e:
            logger.error(f"Error downloading {gbif_id} from {image_url}: {e}")
            continue

    return False


def resize_image(gbif_id, local_path):
    changed, new_size = resize_with_aspect_ratio(local_path, local_path, max_size=1024, format="JPEG", quality=85)
    if changed:
        logger.info(f"Resized {gbif_id} to {new_size}. Path: {local_path}")
    else:
        logger.info(f"Skipped resizing {gbif_id}; already <= target. Path: {local_path}")


def process_id(gbif_id, candidate_urls):
    global n_installed
    gbif_id = str(gbif_id)

    with counter_lock:
        current_total = n_installed
    if current_total % 10000 == 0 and current_total > 0:
        logger.info(f"Checkpointed {current_total} images so far.")

    if gbif_id in processed_ids:
        logger.info(f"{gbif_id} already processed (checkpoint), skipping.")
        return
    if gbif_id in failed_ids:
        logger.info(f"{gbif_id} previously failed, skipping.")
        return

    local_path = get_hierarchical_path(INSTALL_PATH, gbif_id, ".jpg")
    downloaded = False
    existing_valid = False

    if is_duplicate(gbif_id):
        logger.warning(f"Image {gbif_id} is a duplicate from earlier datasets; skipping download.")
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.warning(f"Removed existing file for duplicate {gbif_id} at {local_path}.")
            except Exception as e:
                logger.error(f"Failed removing duplicate file for {gbif_id}: {e}")
        return

    if os.path.exists(local_path):
        logger.info(f"Image {gbif_id} already exists at {local_path}, verifying size...")
        try:
            size = get_file_size_in_mb(local_path)
        except FileNotFoundError:
            size = 0.0

        if size < 0.01:
            logger.warning(f"Image {gbif_id} is too small ({size:.4f} MB), redownloading from alternatives")
            downloaded = download_image_from_candidates(gbif_id, candidate_urls, local_path)
            if downloaded:
                logger.info(f"Successfully re-downloaded {gbif_id}, proceeding to resize.")
        else:
            with checkpoint_lock:
                if gbif_id not in processed_ids:
                    with open(CHECKPOINT_FILE, "a") as f:
                        f.write(gbif_id + "\n"); f.flush(); os.fsync(f.fileno())
                    processed_ids.add(gbif_id)
            return
    else:
        logger.info(f"Attempting {gbif_id} → {local_path} (trying {len(candidate_urls)} source(s))")
        downloaded = download_image_from_candidates(gbif_id, candidate_urls, local_path)
        if downloaded:
            logger.info(f"Successfully downloaded {gbif_id}, proceeding to resize.")

    if downloaded:
        with counter_lock:
            n_installed += 1
            current = n_installed
        if current % 50000 == 0:
            send_notification("Image Installation", f"Installed {current} images. Remaining: {total_to_install - current}")
            logger.info(f"Installed {current} images")
        try:
            resize_image(gbif_id, local_path)
            with checkpoint_lock:
                if gbif_id not in processed_ids:
                    with open(CHECKPOINT_FILE, "a") as f:
                        f.write(gbif_id + "\n"); f.flush(); os.fsync(f.fileno())
                    processed_ids.add(gbif_id)
        except (OSError, UnidentifiedImageError) as e:
            try:
                os.remove(local_path)
            except Exception:
                pass
            logger.error(f"Error resizing {gbif_id}: {e}. File removed.")
            downloaded = False

    if not downloaded:
        logger.warning(f"All download attempts failed for {gbif_id}. Marking as failed.")
        with checkpoint_lock:
            if gbif_id not in failed_ids:
                with open(FAILED_FILE, "a") as f:
                    f.write(gbif_id + "\n"); f.flush(); os.fsync(f.fileno())
                failed_ids.add(gbif_id)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--country", dest="country", help="Country to download samples from", metavar="COUNTRY CODE")
    args = parser.parse_args()
    country = args.country

    cols = ['gbifID','identifier','countryCode']
    df = pd.read_csv(GBIF_MULTIMEDIA_DATA,
                 delimiter="\t",
                 usecols=lambda c: c in cols,
                 on_bad_lines='skip')


    print(f"Length of multimedia.txt (rows): {len(df)}")

    if 'countryCode' in df.columns and country:
        df = df[df['countryCode'] == country]


    grouped = df.groupby('gbifID')['identifier'].apply(list)
    unique_ids = grouped.index.tolist()
    total_to_install = len(unique_ids)
    print(f"Unique gbifIDs to process: {total_to_install}")

    send_notification("Image Installation", f"Starting image installation for {total_to_install} unique images")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_id, gbif_id, grouped.loc[gbif_id]) for gbif_id in unique_ids]

    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Waiting for threads to finish...")
        executor.shutdown(wait=True, cancel_futures=True)
    finally:
        logger.info(f"Final processed IDs: {len(processed_ids)}, failed: {len(failed_ids)}")


    print(f"All done. Number of installed images: {n_installed}")
