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

host_error_counts = {}
HOST_ERROR_THRESHOLD = 500
circuit_breaker_lock = threading.Lock()

"""
Image install script to download images from a GBIF multimedia.txt file.
Updated to download ALL images per gbifID (not just the first successful one).
Each image is saved with a suffix: <gbifID>-00.jpg, <gbifID>-01.jpg, etc.
A gbifID is marked as processed only when ALL its images are successfully downloaded.
Accurate as of March 2026.
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


existing_gbif_ids = set()

print("Duplicate check against old datasets disabled.")

n_installed = 0

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
]

session = req.Session()
retry_strategy = Retry(
    total=2,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def get_hierarchical_path(base_dir, gbif_id, suffix, ext=".jpg"):
    """
    Build a hierarchical storage path to avoid too many files in one directory.
    suffix: image index suffix e.g. '-00', '-01'. Pass '' for single-image compatibility.
    Example: gbifID=1057161997, suffix='-00' -> <base>/105/716/<gbifID>-00.jpg
    """
    stem = str(gbif_id)
    prefix1 = stem[:3] if len(stem) >= 3 else stem
    prefix2 = stem[3:6] if len(stem) >= 6 else "000"
    dest_dir = os.path.join(base_dir, prefix1, prefix2)
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, f"{stem}{suffix}{ext}")


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

def is_host_circuit_broken(url):
    host = _host_from_url(url)
    with circuit_breaker_lock:
        return host_error_counts.get(host, 0) >= HOST_ERROR_THRESHOLD

def increment_host_errors(url, is_rate_limit=False):
    if is_rate_limit:
        return
    host = _host_from_url(url)
    with circuit_breaker_lock:
        host_error_counts[host] = host_error_counts.get(host, 0) + 1
        current_count = host_error_counts[host]
        if current_count == HOST_ERROR_THRESHOLD:
            logger.error(f"CIRCUIT BREAKER ACTIVATED: Host '{host}' reached {HOST_ERROR_THRESHOLD} errors.")
        elif current_count < HOST_ERROR_THRESHOLD:
            logger.warning(f"Host '{host}' error count: {current_count}/{HOST_ERROR_THRESHOLD}")

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
    logger.warning(f"Blocking host '{host}' due to {reason} for ~{int(seconds)}s.")


def is_duplicate(gbif_id):
    return str(gbif_id) in existing_gbif_ids


def extract_image_from_iiif_manifest(manifest_url, gbif_id):
    """
    Fetch a IIIF manifest and extract direct image URLs from it.
    Returns a list of candidate image URLs (multiple resolutions per canvas).
    """
    try:
        response = session.get(
            manifest_url,
            headers={"User-Agent": random.choice(user_agents), "Accept": "application/json"},
            timeout=120,
        )
        if response.status_code != 200:
            logger.warning(f"Failed to fetch IIIF manifest for {gbif_id}: HTTP {response.status_code}")
            return []
        manifest = response.json()
        image_urls = []
        if 'items' in manifest:
            for item in manifest['items']:
                if item.get('type') == 'Canvas' and 'items' in item:
                    for anno_page in item['items']:
                        if anno_page.get('type') == 'AnnotationPage' and 'items' in anno_page:
                            for anno in anno_page['items']:
                                if 'body' in anno:
                                    body = anno['body']
                                    if 'service' in body:
                                        for service in body['service']:
                                            if 'id' in service:
                                                base_url = service['id']
                                                image_urls.append(f"{base_url}/full/1600,/0/default.jpg")
                                                image_urls.append(f"{base_url}/full/1200,/0/default.jpg")
                                                image_urls.append(f"{base_url}/full/800,/0/default.jpg")
        if image_urls:
            logger.info(f"Extracted {len(image_urls)} image URLs from IIIF manifest for {gbif_id}")
        return image_urls
    except Exception as e:
        logger.warning(f"Error parsing IIIF manifest for {gbif_id}: {e}")
        return []


def download_single_image(gbif_id, image_url, local_path):
    """
    Attempt to download one image from image_url and save it to local_path.
    Returns True on success, False on any failure.
    Unlike the old download_image_from_candidates (which stopped at first success),
    this function handles exactly one URL — the caller loops over all URLs.
    """
    if is_host_circuit_broken(image_url):
        logger.info(f"Host circuit broken (>{HOST_ERROR_THRESHOLD} errors); skipping {gbif_id}: {image_url}")
        return False
    if is_host_blocked(image_url):
        logger.info(f"Host under cooldown; skipping {gbif_id}: {image_url}")
        return False

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
            timeout=180,
        )
        status = image_response.status_code

        if status == 429:
            increment_host_errors(image_url, is_rate_limit=True)
            block_host(image_url, image_response.headers.get("Retry-After"))
            del image_response
            return False

        if status >= 500:
            increment_host_errors(image_url)
            logger.error(f"Server error {status} for {gbif_id} from {image_url}")
            del image_response
            return False

        if status != 200:
            increment_host_errors(image_url)
            logger.error(f"HTTP {status} for {gbif_id} from {image_url}")
            del image_response
            return False

        ctype = (image_response.headers.get("Content-Type") or "").lower()
        if ctype and any(bad in ctype for bad in ["text/html", "text/plain", "application/xml"]):
            increment_host_errors(image_url)
            logger.error(f"Invalid content type for {gbif_id} from {image_url}: {ctype}")
            del image_response
            return False

        with open(local_path, "wb") as out_file:
            shutil.copyfileobj(image_response.raw, out_file)

        logger.info(f"Downloaded {gbif_id} suffix based on index -> {local_path}")
        del image_response
        return True

    except (ConnectTimeout, ReadTimeout, Timeout) as e:
        logger.error(f"Timeout for {gbif_id} from {image_url}: {e}")
        block_host(image_url, timeout_issue=True)
        return False
    except Exception as e:
        increment_host_errors(image_url)
        logger.error(f"Unexpected error for {gbif_id} from {image_url}: {e}")
        return False


def resize_image(gbif_id, local_path):
    changed, new_size = resize_with_aspect_ratio(local_path, local_path, max_size=1024, format="JPEG", quality=85)
    if changed:
        logger.info(f"Resized {gbif_id} to {new_size}. Path: {local_path}")
    else:
        logger.info(f"No resize needed for {gbif_id}. Path: {local_path}")


def process_id(gbif_id, candidate_urls):
    """
    Download ALL images associated with a gbifID.
    Each URL is saved as a separate file: <gbifID>-00.jpg, <gbifID>-01.jpg, etc.
    A gbifID is written to processed_ids only if ALL images succeed.
    If any image fails, the gbifID is written to failed_ids for later retry.
    """
    global n_installed
    gbif_id = str(gbif_id)

    # Skip if already fully processed
    if gbif_id in processed_ids:
        return

    # Skip duplicates already present in earlier datasets
    if is_duplicate(gbif_id):
        logger.warning(f"Duplicate gbifID {gbif_id} found in earlier datasets; skipping.")
        return

    # Expand any IIIF manifest URLs into direct image URLs
    expanded_urls = []
    for url in candidate_urls:
        if '/manifest' in url or url.endswith('.json'):
            extracted = extract_image_from_iiif_manifest(url, gbif_id)
            if extracted:
                expanded_urls.extend(extracted)
            # Do not add the manifest URL itself as a download target
        else:
            expanded_urls.append(url)

    if not expanded_urls:
        logger.warning(f"No downloadable URLs found for {gbif_id}")
        with checkpoint_lock:
            if gbif_id not in failed_ids:
                with open(FAILED_FILE, "a") as f:
                    f.write(gbif_id + "\n"); f.flush(); os.fsync(f.fileno())
                failed_ids.add(gbif_id)
        return

    # Deduplicate URLs while preserving order
    seen = set()
    unique_urls = []
    for url in expanded_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    logger.info(f"Processing {gbif_id}: {len(unique_urls)} unique image URL(s)")

    all_success = True
    downloaded_count = 0

    for idx, image_url in enumerate(unique_urls):
        suffix = f"-{idx:02d}"  # e.g. -00, -01, -02
        local_path = get_hierarchical_path(INSTALL_PATH, gbif_id, suffix, ".jpg")

        # Skip this image if it already exists and is valid
        if os.path.exists(local_path):
            try:
                size = get_file_size_in_mb(local_path)
            except FileNotFoundError:
                size = 0.0
            if size >= 0.01:
                logger.info(f"Already exists and valid: {local_path}, skipping.")
                downloaded_count += 1
                continue
            else:
                logger.warning(f"Existing file too small ({size:.4f} MB), re-downloading: {local_path}")

        success = download_single_image(gbif_id, image_url, local_path)

        if success:
            try:
                resize_image(gbif_id, local_path)
                downloaded_count += 1
                with counter_lock:
                    n_installed += 1
                    current = n_installed
                if current % 50000 == 0:
                    send_notification("Image Installation",
                                      f"Installed {current} images. Remaining: {total_to_install - current}")
                    logger.info(f"Installed {current} images total")
            except (OSError, UnidentifiedImageError) as e:
                try:
                    os.remove(local_path)
                except Exception:
                    pass
                logger.error(f"Resize failed for {gbif_id} index={idx}: {e}. File removed.")
                all_success = False
        else:
            logger.warning(f"Failed to download {gbif_id} index={idx} from {image_url}")
            all_success = False

    # Write to processed_ids only if every image succeeded
    # Otherwise write to failed_ids for retry
    with checkpoint_lock:
        if all_success and downloaded_count == len(unique_urls):
            if gbif_id not in processed_ids:
                with open(CHECKPOINT_FILE, "a") as f:
                    f.write(gbif_id + "\n"); f.flush(); os.fsync(f.fileno())
                processed_ids.add(gbif_id)
                # Remove from failed_ids if it was a retry that succeeded
                if gbif_id in failed_ids:
                    failed_ids.discard(gbif_id)
                logger.info(f"All {downloaded_count} image(s) done for {gbif_id}. Marked as processed.")
        else:
            if gbif_id not in failed_ids:
                with open(FAILED_FILE, "a") as f:
                    f.write(gbif_id + "\n"); f.flush(); os.fsync(f.fileno())
                failed_ids.add(gbif_id)
                logger.warning(f"Partial failure for {gbif_id}: {downloaded_count}/{len(unique_urls)} downloaded.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--country", dest="country", help="Country to download samples from", metavar="COUNTRY CODE")
    args = parser.parse_args()
    country = args.country

    cols = ['gbifID', 'identifier', 'countryCode']
    df = pd.read_csv(GBIF_MULTIMEDIA_DATA,
                     delimiter="\t",
                     usecols=lambda c: c in cols,
                     on_bad_lines='skip')
    print(f"Length of multimedia.txt (rows): {len(df)}")
    
    if 'countryCode' in df.columns and country:
        df = df[df['countryCode'] == country]
    
    grouped = df.groupby('gbifID')['identifier'].apply(list)
    unique_ids = grouped.index.tolist()
    
    if failed_ids:
        failed_ids_int = {int(gid) for gid in failed_ids if gid.isdigit()}
        grouped_keys = set(grouped.index)
        retry_ids = [gid for gid in failed_ids_int if gid in grouped_keys]
        if retry_ids:
            print(f"Retrying {len(retry_ids)} previously failed IDs first...")
            # NOTE: Do NOT clear failed_ids here. Each ID will be removed
            # from failed_ids only after it successfully downloads, in process_id().
            unique_ids = retry_ids
    total_to_install = len(unique_ids)
    print(f"Unique gbifIDs to process: {total_to_install}")

    send_notification("Image Installation", f"Starting image installation for {total_to_install} unique gbifIDs")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_id, gbif_id, grouped.loc[gbif_id]) for gbif_id in unique_ids]

    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Waiting for threads to finish...")
        executor.shutdown(wait=True, cancel_futures=True)
    finally:
        with open(FAILED_FILE, "w") as f:
            for fid in failed_ids:
                f.write(str(fid) + "\n")
            f.flush(); os.fsync(f.fileno())
        logger.info(f"Final processed IDs: {len(processed_ids)}, failed: {len(failed_ids)}")
        logger.info(f"Circuit breaker status: {len([h for h, c in host_error_counts.items() if c >= HOST_ERROR_THRESHOLD])} hosts permanently blocked")
        for host, count in sorted(host_error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {host}: {count} errors")

    print(f"All done. Number of installed images: {n_installed}")