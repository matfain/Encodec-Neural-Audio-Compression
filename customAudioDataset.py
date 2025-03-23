import os
import random
import librosa
import pandas as pd
import torch
import audioread
import logging

logger = logging.getLogger(__name__)

from utils import convert_audio

def remove_first_n_dirs(path, n=1):
    """
    Removes the first n directory levels from the given path.
    For example, if path is:
      "./parsed_downloads/LibriSpeech360/LibriTTS_R/train-clean-360/..."
    then remove_first_n_dirs(path, n=1) will return:
      "LibriSpeech360/LibriTTS_R/train-clean-360/..."
    """
    normalized = os.path.normpath(path)  # e.g. "parsed_downloads/LibriSpeech360/LibriTTS_R/..."
    parts = normalized.split(os.sep)
    if len(parts) <= n:
        return normalized
    return os.path.join(*parts[n:])

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None, mode='train'):
        """
        mode: either 'train' or 'test'
        """
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        
        # Read CSV path from config
        csv_path = (
            config.datasets.train_csv_path if mode == 'train'
            else config.datasets.test_csv_path
        )
        self.audio_files = pd.read_csv(csv_path, on_bad_lines='skip')
        
        # Move one directory up from the CSV's folder to get to "parsed_downloads"
        # Example: if CSV is at ".../parsed_downloads/dataset_all_Librispeech/train.csv",
        # then base_path becomes ".../parsed_downloads".
        self.base_path = os.path.dirname(os.path.dirname(csv_path))
        
        self.transform = transform
        self.fixed_length = config.datasets.fixed_length
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels

    def __len__(self):
        # If fixed_length is set and smaller than total number of files, limit length
        if self.fixed_length and len(self.audio_files) > self.fixed_length:
            return self.fixed_length
        return len(self.audio_files)

    def get(self, idx=None):
        """Uncropped, untransformed getter with random sample feature."""
        if idx is not None and idx >= len(self.audio_files):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        
        # Read the path from CSV (e.g. "./parsed_downloads/LibriSpeech360/LibriTTS_R/..."),
        # then remove the first directory ("parsed_downloads"), leaving "LibriSpeech360/LibriTTS_R/...".
        relative_path = self.audio_files.iloc[idx, 0]
        relative_path = remove_first_n_dirs(relative_path, n=1)
        
        # Join with self.base_path (e.g. ".../datasets/parsed_downloads + LibriSpeech360/LibriTTS_R/...").
        file_path = os.path.abspath(os.path.join(self.base_path, relative_path))
        logger.debug(f'Loading audio file: {file_path}')
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            # Attempt to load another sample instead of crashing
            return self[idx]
        
        try:
            waveform, sample_rate = librosa.load(
                file_path, 
                sr=self.sample_rate, 
                mono=(self.channels == 1)
            )
        except (audioread.exceptions.NoBackendError, ZeroDivisionError):
            logger.warning(f"Not able to load {relative_path}, removing from dataset")
            self.audio_files.drop(idx, inplace=True)
            return self[idx]

        # Convert waveform to tensor and adjust dimensions if mono
        waveform = torch.as_tensor(waveform)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # shape (1, num_samples)
            waveform = waveform.expand(self.channels, -1)  # shape (channels, num_samples)

        return waveform, sample_rate

    def __getitem__(self, idx):
        waveform, sample_rate = self.get(idx)
        
        # Safety check if get() fails repeatedly
        if waveform is None:
            raise RuntimeError(f"Failed to load a valid audio sample at index {idx}")
        
        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1] - self.tensor_cut - 1)
                waveform = waveform[:, start:start + self.tensor_cut]
                return waveform, sample_rate
            else:
                return waveform, sample_rate
        
        return waveform, sample_rate

def pad_sequence(batch):
    """
    Make all tensors in a batch the same length by padding with zeros.
    """
    batch = [item.permute(1, 0) for item in batch]  # (channels, length) -> (length, channels)
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)  # (batch, length, channels) -> (batch, channels, length)
    return batch

def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    tensors = []
    for waveform, _ in batch:
        tensors.append(waveform)
    tensors = pad_sequence(tensors)
    return tensors
