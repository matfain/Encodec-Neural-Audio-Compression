#!/usr/bin/env python3

import os
import pandas as pd
from pathlib import Path

def gather_files(root_dir, extensions=('.flac', '.wav', '.mp3')):
    """
    Recursively gather audio files from the specified root_dir.
    """
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(extensions):
                file_list.append(os.path.join(root, f))
    return file_list

def split_train_val_test(file_list, output_dir, train_ratio=0.95, val_ratio=0.025, random_state=42):
    """
    Shuffle and split the combined file list into train, validation, and test CSV files without headers.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("Please install scikit-learn: pip install scikit-learn")

    # Convert file list to a DataFrame
    data = pd.DataFrame(file_list, columns=["filepath"])
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate test ratio (remaining data)
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # First split: train vs. (validation+test)
    train_data, val_test_data = train_test_split(
        data, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # Second split: validation vs. test
    val_data, test_data = train_test_split(
        val_test_data,
        train_size=val_ratio / (val_ratio + test_ratio),
        random_state=random_state
    )
    
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write CSV files without headers
    train_data.to_csv(output_dir / "train.csv", index=False, header=False)
    val_data.to_csv(output_dir / "validation.csv", index=False, header=False)
    test_data.to_csv(output_dir / "test.csv", index=False, header=False)
    
    print(f"Total files: {len(file_list)}")
    print(f"Train split: {len(train_data)}")
    print(f"Validation split: {len(val_data)}")
    print(f"Test split: {len(test_data)}")

def main():
    """
    Hardcode the folders you want to include and the output directory below.
    Just update these lists/variables whenever you need a different selection.
    """
    # 1) List the folders you want to include.
    #    For example, if you only want to use LibriSpeech and LibriSpeech360:
    data_folders = [
        "./parsed_downloads/FMA_Small",
        "./parsed_downloads/LibriSpeech100"
        # Add more files as you wish
    ]
    
    # 2) Set your output directory where train.csv, validation.csv, and test.csv will be saved:
    output_dir = "./parsed_downloads/dataset_libri100_FMA_small"
    
    # 3) Adjust these as desired:
    train_ratio = 0.9
    val_ratio = 0.05
    random_state = 42
    
    # Gather all files from the chosen folders
    file_list = []
    for folder in data_folders:
        file_list.extend(gather_files(folder))
    
    # Split into train, validation, and test
    split_train_val_test(
        file_list=file_list,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state
    )

if __name__ == "__main__":
    main()
