import os
import tarfile
import zipfile
import subprocess
import shutil

FMA_SMALL = "https://tile.loc.gov/storage-services/master/gdc/gdcdatasets/2018655052_small/2018655052_small.zip"
FMA_MEDIUM = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"

def download_data_with_wget(url, output_name, extract_to):
    if not url or not output_name:
        print("[ERROR] Bad download request: must supply download URL and output name.")
        return

    # Ensure wget is installed
    if shutil.which("wget") is None:
        print("[ERROR] wget is not installed. Please install wget and try again.")
        return

    # Replace domain with resolved IP if necessary
    ip_address = "86.119.28.16"  # Use the IP from nslookup
    modified_url = url.replace("os.unil.cloud.switch.ch", ip_address)

    download_dir = os.path.join(os.getcwd(), 'raw_downloads')
    os.makedirs(download_dir, exist_ok=True)
    
    output_path = os.path.join(download_dir, output_name)
    
    print(f"[INFO] Starting download: {modified_url}")
    print(f"[INFO] Saving to: {output_path}")

    try:
        # Use wget with -c (resume) and --header to specify the correct hostname
        subprocess.run(["wget", "-c", "--no-check-certificate", "--header=Host: os.unil.cloud.switch.ch", "-O", output_path, modified_url], check=True)

        # Verify file was downloaded
        if not os.path.exists(output_path):
            print(f"[ERROR] Download failed, file not found: {output_path}")
            return
        
        print(f"[SUCCESS] Download complete! File saved at: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {modified_url}: {e}")
        return
    
    # Extract if necessary
    if output_path.endswith(".tar.gz"):
        extract_tar_gz(output_path, extract_to)
    elif output_path.endswith(".zip"):
        extract_zip(output_path, extract_to)

def extract_tar_gz(extract_from, extract_to):
    if not os.path.exists(extract_from):
        print(f"[ERROR] File not found: {extract_from}")
        return
    
    print(f"[INFO] Extracting {extract_from} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    with tarfile.open(extract_from, "r:gz") as tar:
        tar.extractall(path=extract_to)
    
    print(f"[SUCCESS] Extraction complete: Files are in {extract_to}")

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"[ERROR] File not found: {zip_path}")
        return

    print(f"[INFO] Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"[SUCCESS] Extraction complete: Files are in {extract_to}")

if __name__ == "__main__":
    extract_to = os.path.join(os.getcwd(), "parsed_downloads/FMA_Medium")
    download_data_with_wget(url=FMA_MEDIUM, output_name="fma_medium.zip", extract_to=extract_to)
