#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import pandas as pd

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
    parser = argparse.ArgumentParser(
        description="Generic audio dataset splitter with default arguments and no CSV headers."
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='./parsed_downloads',
        help="Root directory containing all subfolders with audio files. (Default: './parsed_downloads')"
    )
    # Output directory is set to the same as the root_dir by default.
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./parsed_downloads',
        help="Directory to store the resulting train/validation/test CSV files. (Default: same as root_dir)"
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.95,
        help="Train set ratio. (Default: 0.95)"
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.025,
        help="Validation set ratio. (Default: 0.025)"
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help="Random seed for reproducible shuffling. (Default: 42)"
    )
    
    args = parser.parse_args()
    
    # Gather all audio files under the root_dir
    file_list = gather_files(args.root_dir)
    
    # Split the files into train, validation, and test CSVs placed in the output_dir
    split_train_val_test(
        file_list=file_list,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()
