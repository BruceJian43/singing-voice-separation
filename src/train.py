import argparse
import torch
import torch.nn
import torchaudio

import models
import preprocess
import utils

from pathlib import Path
from dataset import SpeechSeparationDataset

from tqdm import tqdm

def train(dataset_path: str,
          target_channel: int,
          n_fft: int = 2048,
          win_length: int = 2048,
          hop_length: int = 512,
          sample_rate: int = 16000,
          batch_size: int = 64,
          learning_rate: float = 3e-4,
          n_frame_in_segment: int = 15,
          n_epochs: int = 100,
          model_path: str = 'model.ckpt',
          seed: int = 42,
          model_name: str = 'DAE') -> None:

    utils.same_seeds(seed)
    device = utils.get_device()
    print(f'device: {device}')
    print(f'model name: {model_name}')
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Not a correct directory path.')
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f'Not a directory')

    train_dataset = SpeechSeparationDataset(dataset_dir / 'train', target_channel, n_frame_in_segment, n_fft, win_length, hop_length, sample_rate)
    val_dataset = SpeechSeparationDataset(dataset_dir / 'val', target_channel, n_frame_in_segment, n_fft, win_length, hop_length, sample_rate)

    model = getattr(models, model_name)()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        training_loss_list = []
        for x, y in tqdm(train_dataloader):
            y_hat = model(x.to(device))
            loss = criterion(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss_list.append(loss.item())
        training_loss = sum(training_loss_list) / len(training_loss_list)

        model.eval()
        val_loss_list = []
        for x, y in  tqdm(val_dataloader):
            with torch.no_grad():
                y_hat = model(x.to(device))
            loss = criterion(y_hat, y.to(device))
            val_loss_list.append(loss.item())
        val_loss = sum(val_loss_list) / len(val_loss_list)

        print(f'epoch: {epoch + 1}/{n_epochs}, training_loss = {training_loss:.5f}, validation_loss = {val_loss:.5f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-path', type=str, required=True, help='path of the dataset')
    parser.add_argument('--target-channel', type=int, required=True, help='index of the channel to be separated')
    parser.add_argument('--n-fft', type=int, default=2048, help='number of fft (argument for stft)')
    parser.add_argument('--win-length', type=int, default=2048, help='window length (argument for stft)')
    parser.add_argument('--hop-length', type=int, default=512, help='hop length (argument for stft)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='sample rate to resample input wav file')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size (argument for training)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='learning rate (argument for training)')
    parser.add_argument('--n-epochs', type=int, default=100, help='number of epochs (argument for training)')
    parser.add_argument('--n-frame-in-segment', type=int, default=15, help='number of frames of spectrogram of a 2D segment (argument for training) ')
    parser.add_argument('--model-path', type=str, default='model.ckpt', help='path to save trained model')
    parser.add_argument('--seed', type=int, default=42, help='random seed for training')
    parser.add_argument('--model-name', type=str, default='DAE', help='selected class in model.py')
    args = vars(parser.parse_args())
    train(**args)