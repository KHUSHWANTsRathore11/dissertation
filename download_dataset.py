import os
import zipfile
import argparse

def download_kaggle_dataset(dataset_name, download_path="dataset"):
    # Ensure Kaggle credentials are set or present in ~/.kaggle/kaggle.json
    print(f"Downloading dataset {dataset_name}...")
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    ret = os.system(f"kaggle datasets download -d {dataset_name} -p {download_path}")
    if ret != 0:
        print("Download failed. Make sure you have kaggle credentials configured in ~/.kaggle/kaggle.json")
        return
    
    zip_file = f"{download_path}/{dataset_name.split('/')[1]}.zip"
    if os.path.exists(zip_file):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print("Extraction complete.")
        os.remove(zip_file)
    else:
        print("Download zip not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="souradeep99/wire-rope-abrasion", help="Kaggle dataset name")
    parser.add_argument("--path", type=str, default="dataset", help="Download path")
    args = parser.parse_args()
    download_kaggle_dataset(args.dataset, args.path)
