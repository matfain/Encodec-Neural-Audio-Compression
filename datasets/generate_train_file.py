import os
import argparse
from pathlib import Path
import pandas as pd  

def generate_csv(file_dir, csv_path,mode='train'):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (file.endswith('.flac') or file.endswith('.wav') or file.endswith('.mp3')) and mode in root:
                file_list.append(os.path.join(root, file))

    print(f"file length:{len(file_list)}")
    csv_path = Path(csv_path)
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True)
        
    data = pd.DataFrame(file_list)
    data.to_csv(csv_path, index=False, header=False)

def split_train_val_test_csv(csv_path, train_ratio=0.8, val_ratio=0.1):
    test_ratio = 1 - train_ratio - val_ratio
    print(f"Starting dataset split: Train = {train_ratio*100:.1f}%, Validation = {val_ratio*100:.1f}%, Test = {test_ratio*100:.1f}%")
    try:
        from sklearn.model_selection import train_test_split  
    except ImportError as E:
        print("Please install required modules with: pip install pandas scikit-learn")
        
    data = pd.read_csv(csv_path)  
    train_data, test_and_val_data = train_test_split(data, train_size=train_ratio, random_state=42)
    test_data, validation_data = train_test_split(test_and_val_data, train_size= test_ratio / (test_ratio + val_ratio), random_state=42)  

    base_path = Path(csv_path).with_suffix('')  # Removes the suffix from the path

    train_data.to_csv(f'{base_path}_train.csv', index=False)
    validation_data.to_csv(f'{base_path}_validation.csv', index=False)
    test_data.to_csv(f'{base_path}_test.csv', index=False)

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('-i','--input_file_dir', type=str, default='./parsed_downloads/LibriSpeech100/LibriSpeech/train-clean-100')
    arg.add_argument('-o','--output_path', type=str, default='./parsed_downloads/librispeech_train100h.csv')
    arg.add_argument('-m','--mode', type=str, default='train',help='train,test-clean/other or dev-clean/other')
    arg.add_argument('-s','--split', action='store_true', default=False,help='Split dataset into train/test')
    arg.add_argument('-t','--train_threshold',type=float,default=0.8)
    arg.add_argument('-v','--validation_threshold',type=float,default=0.1)

    args = arg.parse_args()
    generate_csv(args.input_file_dir, args.output_path,args.mode)
    if args.split:
        split_train_val_test_csv(args.output_path,train_ratio=args.train_threshold, val_ratio=args.validation_threshold)