import argparse
import torch
import torch.nn
import torchaudio

import preprocess
import utils

from pathlib import Path


def inference_an_input(model: torch.nn.Module,
                       in_wav: Path,
                       n_frame_in_segment: int,
                       n_fft: int,
                       win_length: int,
                       hop_length: int,
                       sample_rate: int,
                       batch_size: int,
                       resample: bool,
                       out_dir: Path = Path('output'),
                       save_file: bool = True) -> torch.Tensor:
    in_wav = Path(in_wav)
    out_dir = Path(out_dir)
    if save_file:
        out_dir.mkdir(parents=True, exist_ok=True)
    if not in_wav.exists():
        raise FileNotFoundError(f'Not a correct input file path')

    device = utils.get_device()
    model.to(device)
    model.eval()

    spectrogram, time_domain_padded_size = preprocess.convert_to_spectrogram(
        in_wav, n_fft, win_length, hop_length, sample_rate, True, resample)
    magnitude, phase = torch.abs(spectrogram), torch.angle(spectrogram)
    data, freq_domain_padded_size = preprocess.split_tensors_with_padding(
        magnitude, n_frame_in_segment)

    pred = []
    start_idx, end_idx = 0, batch_size
    while start_idx < data.shape[0]:
        x = data[start_idx:end_idx]
        with torch.no_grad():
            y = model(x.to(device))
        pred.append(y.cpu())

        start_idx += batch_size
        end_idx += batch_size
        if end_idx > data.shape[0]:
            end_idx = data.shape[0]
    pred = torch.cat(pred, dim=0)
    pred_padded_spectrogram = torch.cat(torch.unbind(pred), dim=-1)
    pred_spectrogram = pred_padded_spectrogram if freq_domain_padded_size == 0 else pred_padded_spectrogram[:, :, 0: -freq_domain_padded_size]

    istft = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    pred_padded_waveform = istft(torch.polar(pred_spectrogram, phase))
    pred_waveform = pred_padded_waveform if time_domain_padded_size == 0 else pred_padded_waveform[:, 0:-time_domain_padded_size]
    if save_file:
        torchaudio.save(out_dir / in_wav.name, pred_waveform, sample_rate)

    return pred_waveform.cpu()


def test(model_path: str,
         in_dir: str,
         out_dir: str,
         n_frame_in_segment: int,
         n_fft: int,
         win_length: int,
         hop_length: int,
         sample_rate: int,
         batch_size: int,
         resample: bool) -> None:

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f'Not a correct directory path.')
    if not in_dir.is_dir():
        raise NotADirectoryError(f'Not a directory')
    out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.load(model_path)

    for f in in_dir.iterdir():
        if not utils.is_extension_supported(f):
            continue
        inference_an_input(model, f, n_frame_in_segment, n_fft,
                           win_length, hop_length, sample_rate, batch_size, resample, out_dir, save_file=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str, required=True, help='path of model used for the inference')
    parser.add_argument('--in-dir', type=str, required=True, help='path of the input dir')
    parser.add_argument('--out-dir', type=str, default='output', help='path of the output dir')
    parser.add_argument('--n-fft', type=int, default=2048, help='number of fft (argument for stft)')
    parser.add_argument('--win-length', type=int, default=2048, help='window length (argument for stft)')
    parser.add_argument('--hop-length', type=int, default=512, help='hop length (argument for stft)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='sample rate to resample input wav file')
    parser.add_argument('--n-frame-in-segment', type=int, default=15, help='number of frames of spectrogram of a 2D segment')
    parser.add_argument('--batch-size', type=int, default=64, help='number of segments in a batch')
    parser.add_argument('--resample', type=bool, default=True, help='to resample input or not')
    args = vars(parser.parse_args())
    test(**args)
