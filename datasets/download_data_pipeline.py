import requests
import os
import time
import tarfile
import zipfile

LIBRISPEECH_100_TRAIN = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_360_TRAIN = "https://www.openslr.org/resources/141/train_clean_360.tar.gz"
LIBRISPEECH_500_TRAIN = "https://www.openslr.org/resources/12/train-other-500.tar.gz"

FSD_50K = "https://zenodo.org/api/records/4060432/files-archive"

FMA_SMALL = "https://tile.loc.gov/storage-services/master/gdc/gdcdatasets/2018655052_small/2018655052_small.zip" # Alternate link that can be derived from here https://www.loc.gov/item/2018655052/?loclr=blogsig

def download_data_from_url_to_output_folder(url, output_name, extract_folder):
    if not url or not output_name:
        print("[ERROR] Bad download request: must supply download URL and output name.")
        return

    download_dir = os.path.join(os.getcwd(), 'raw_downloads')
    os.makedirs(download_dir, exist_ok=True)
    
    output_path = os.path.join(download_dir, output_name)
    
    print(f"[INFO] Starting download: {url}")
    print(f"[INFO] Saving to: {output_path}")
    
    # Stream the download with progress tracking
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise an error if the request fails
        
        total_size = int(response.headers.get('Content-Length', 0))  # Get total file size
        chunk_size = 8192
        num_chunks = total_size // chunk_size if total_size > 0 else None
        
        downloaded_size = 0
        start_time = time.time()
        
        with open(output_path, "wb") as file:
            for i, chunk in enumerate(response.iter_content(chunk_size=chunk_size)):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Print progress every 5MB downloaded
                    if i % (5 * 1024 * 1024 // chunk_size) == 0:
                        percent = (downloaded_size / total_size * 100) if total_size else 0
                        elapsed_time = time.time() - start_time
                        speed = (downloaded_size / 1024 / 1024) / elapsed_time if elapsed_time > 0 else 0
                        print(f"[INFO] Downloaded: {downloaded_size / (1024*1024):.2f}MB ({percent:.2f}% complete) - {speed:.2f} MB/s")
    
    print(f"[SUCCESS] Download complete! File saved at: {output_path}")
    
    if output_path.endswith(".tar.gz"):
        extract_tar_gz(output_path, extract_folder)
    elif output_path.endswith(".zip"):
        extract_zip(output_path, extract_folder)


def extract_tar_gz(extract_from, extract_to):
    """
    Extracts a .tar.gz archive to a specified directory.

    :param archive_path: Path to the .tar.gz file
    :param extract_to: Directory where files will be extracted
    """
    if not os.path.exists(extract_from):
        print(f"[ERROR] File not found: {extract_from}")
        return
    
    print(f"[INFO] Extracting {extract_from} to {extract_to}...")
    
    os.makedirs(extract_to, exist_ok=True)  # Ensure directory exists
    
    with tarfile.open(extract_from, "r:gz") as tar:
        tar.extractall(path=extract_to)
    
    print(f"[SUCCESS] Extraction complete: Files are in {extract_to}")
    

def extract_zip(zip_path, extract_to):
    """
    Extracts a .zip archive to a specified directory.

    :param zip_path: Path to the .zip file
    :param extract_to: Directory where files will be extracted
    """
    if not os.path.exists(zip_path):
        print(f"[ERROR] File not found: {zip_path}")
        return

    print(f"[INFO] Extracting {zip_path} to {extract_to}...")

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"[SUCCESS] Extraction complete: Files are in {extract_to}")


if __name__ == "__main__":
    # extract_to = os.path.join(os.getcwd(), "parsed_downloads/LibriSpeech100")
    # download_data_from_url_to_output_folder(url=LIBRISPEECH_100_TRAIN, output_name="librispeech_100_train.tar.gz")
    
    # extract_to = os.path.join(os.getcwd(), "parsed_downloads/FMA_Small")
    # download_data_from_url_to_output_folder(url=FMA_SMALL, output_name="fma_small.zip", extract_folder=extract_to)
    
    # extract_to = os.path.join(os.getcwd(), "parsed_downloads/LibriSpeech360")
    # download_data_from_url_to_output_folder(url=LIBRISPEECH_360_TRAIN, output_name="librispeech_360_train.tar.gz", extract_folder=extract_to)
    
    extract_to = os.path.join(os.getcwd(), "parsed_downloads/FSD_50K")
    download_data_from_url_to_output_folder(url=FSD_50K, output_name="fsd_50k.zip", extract_folder=extract_to)
