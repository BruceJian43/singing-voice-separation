import torch.nn
import torchaudio
import random
import shutil

import utils

from pathlib import Path

import argparse

def mix_channels(in_wav: torch.Tensor) -> torch.Tensor:
    if in_wav.dim() > 2:
        raise ValueError(f'incorrect wavform shape for channel mixing.')
    elif in_wav.dim() == 1:
        out_wav = in_wav
    else:
        out_wav = torch.mean(in_wav, 0, keepdim=True)
    return out_wav

def resample_wav(in_wav: torch.Tensor, orig_sample_rate: int, new_sample_rate: int) -> torch.Tensor:
    resampler = torchaudio.transforms.Resample(orig_sample_rate, new_sample_rate)
    out_wav = resampler(in_wav)
    return out_wav

def convert_to_spectrogram(in_file: Path,
                           n_fft: int = 2048,
                           win_length: int = 2048,
                           hop_length: int = 512,
                           sample_rate: int = 16000,
                           mono=False,
                           resample=True) -> tuple[torch.Tensor, int]:
    if not in_file.exists():
        raise FileNotFoundError(f'Not a correct file path')
    waveform, orig_sample_rate = torchaudio.load(in_file)
    if not resample:
        sample_rate = orig_sample_rate
    resampled_waveform = resample_wav(waveform, orig_sample_rate, sample_rate)
    n_sample_points = resampled_waveform.shape[1]
    pad_size = (max(0, n_sample_points - win_length) // hop_length +
                1) * hop_length - n_sample_points + win_length
    padded_resampled_waveform = torch.nn.functional.pad(resampled_waveform,
                                                       (0, pad_size),
                                                       mode='constant',
                                                       value=0.0)
    stft = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             power=None,
                                             return_complex=True)
    if mono:
        padded_resampled_waveform = mix_channels(padded_resampled_waveform)
    spectrogram = stft(padded_resampled_waveform)
    return (spectrogram, pad_size)

def split_tensors_with_padding(x: torch.Tensor, split_size: int) -> tuple[torch.Tensor, int]:
    segments = list(torch.split(x, split_size, dim=-1))
    padded_size = 0
    if segments[-1].shape[-1] != split_size:
        padded_size = split_size - segments[-1].shape[-1]
        segments[-1] = torch.nn.functional.pad(segments[-1], (0, padded_size))
    ret = torch.stack(segments)
    return ret, padded_size

def convert_to_batch_data(dataset_path: Path, target_channel: int, n_frame_in_segment: int, n_fft: int = 2048, win_length: int = 2048, hop_length: int = 512, sample_rate: int = 16000):
    """Convert songs to traning set.
    Returns:
        (batch_data_mixed, batch_data_separated)
        batch_data_mixed: spectrograms with a sigle channel (used for training data)
        batch_data_separated: spectrograms with separared channels (used for training ground truth)
    """
    wav_files = [f for f in dataset_path.iterdir() if utils.is_extension_supported(f)]
    mixed_spectrograms = [convert_to_spectrogram(f, n_fft, win_length, hop_length, sample_rate, True)[0] for f in wav_files]
    separated_spectrograms = [convert_to_spectrogram(f, n_fft, win_length, hop_length, sample_rate, False)[0][target_channel].unsqueeze(0) for f in wav_files]
    mixed_magnitudes = [torch.abs(spectrogram) for spectrogram in mixed_spectrograms]
    separated_magnitudes = [torch.abs(spectrogram) for spectrogram in separated_spectrograms]

    batch_data_mixed = torch.cat([split_tensors_with_padding(mag, n_frame_in_segment)[0] for mag in mixed_magnitudes])
    batch_data_separated = torch.cat([split_tensors_with_padding(mag, n_frame_in_segment)[0] for mag in separated_magnitudes])

    return (batch_data_mixed, batch_data_separated)

def split_dataset(in_dir: str, train_dir: str, val_dir: str, test_dir: str, ratio: list = (.8, .1, .1), seed: int = 42) -> None:
    in_dir = Path(in_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    test_dir = Path(test_dir)
    if type(ratio) is not tuple or len(ratio) != 3 or sum(ratio) > 1 or sum(1 for r in ratio if r < 0):
        raise ValueError(f'Invalid ratio to split dataset.')
    if not in_dir.exists():
        raise FileNotFoundError(f'Not a correct directory path.')
    if not in_dir.is_dir():
        raise NotADirectoryError(f'Not a directory')
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    wav_files = [f for f in in_dir.iterdir() if utils.is_extension_supported(f)]
    random.seed(seed)
    random.shuffle(wav_files)
    train_size = int(len(wav_files) * ratio[0])
    val_size = int(len(wav_files) * ratio[1])
    test_size = len(wav_files) - train_size - val_size
    assert train_size >= 0 and val_size >= 0 and test_size >= 0
    for f in wav_files[0:train_size]:
        shutil.copy(in_dir / f.name, train_dir / f.name)
    for f in wav_files[train_size:train_size + val_size]:
        shutil.copy(in_dir / f.name, val_dir / f.name)
    for f in wav_files[train_size + val_size:]:
        shutil.copy(in_dir / f.name, test_dir / f.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-dir', type=str, required=True, help='path of the dataset to split')
    parser.add_argument('--train-dir', type=str, default='./dataset/train')
    parser.add_argument('--val-dir', type=str, default='./dataset/val')
    parser.add_argument('--test-dir', type=str, default='./dataset/test')
    args = vars(parser.parse_args())
    split_dataset(**args)
