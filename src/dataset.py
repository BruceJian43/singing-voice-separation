import torch
import torch.nn as nn
import torchaudio

from preprocess import convert_to_spectrogram, convert_to_batch_data

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class SpeechSeparationDataset(Dataset):
    def __init__(self, dataset_path: str, target_channel: int, n_frame_in_segment: int, n_fft: int = 2048, win_length: int = 2048, hop_length: int = 512, sample_rate: int = 16000):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f'Not a correct dataset path')
        self.target_channel = target_channel
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.x, self.y = convert_to_batch_data(self.dataset_path, target_channel, n_frame_in_segment, n_fft, win_length, hop_length, sample_rate)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]