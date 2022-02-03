import argparse
import torch
import torch.nn
import torchaudio
import numpy

import preprocess
import utils

from test import inference_an_input

from pathlib import Path
from mir_eval.separation import bss_eval_sources

def evaluate(model_paths: list[str], in_dir: str, n_fft: int, win_length: int,
             hop_length: int, sample_rate: int, n_frame_in_segment: int,
             batch_size: int, resample: bool) -> None:
    """Using the same matircs as https://www.music-ir.org/mirex/wiki/2019:Singing_Voice_Separation
    Args:
        model_paths: use model_path[i] to separate ith channel
    """
    models = [utils.load_model(model_path) for model_path in model_paths]
    n_models = len(models)
    accum_NSDR, accum_SIR, accum_SAR, n_files = numpy.zeros(n_models), numpy.zeros(n_models), numpy.zeros(n_models), 0
    in_dir = Path(in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f'Not a correct directory path')
    for f in in_dir.iterdir():
        if not utils.is_extension_supported(f):
            continue
        ground_truth, orig_sample_rate = torchaudio.load(f)
        if resample:
            ground_truth = preprocess.resample_wav(ground_truth, orig_sample_rate, sample_rate)
        n_channels = ground_truth.shape[0]
        if n_channels != n_models:
            continue

        preds = [
            inference_an_input(model,
                               f,
                               n_frame_in_segment,
                               n_fft,
                               win_length,
                               hop_length,
                               sample_rate,
                               batch_size,
                               resample,
                               save_file=False).squeeze(0).numpy() for model in models
        ]

        mono_ground_truth = preprocess.mix_channels(ground_truth).squeeze(0).numpy()
        ground_truth = ground_truth.numpy()
        SDR, SIR, SAR, _ = bss_eval_sources(ground_truth, numpy.array(preds))
        NSDR, _, _, _ = bss_eval_sources(ground_truth, numpy.array([mono_ground_truth, mono_ground_truth]))

        NSDR = SDR - NSDR
        accum_NSDR += NSDR
        accum_SIR += SIR
        accum_SAR += SAR
        n_files += 1
    GNSDR = accum_NSDR / n_files
    GSIR = accum_SIR / n_files
    GSAR = accum_SAR / n_files
    for i in range(n_models):
        print(f'Channel {i}:')
        print(f'GNSDR: {GNSDR[i]:.4f}')
        print(f'GSIR: {GSIR[i]:.4f}')
        print(f'GSAR: {GSAR[i]:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-paths', type=str, required=True, nargs='+', help='paths of models used for the inference for each channel')
    parser.add_argument('--in-dir', type=str, required=True, help='path of the input dir')
    parser.add_argument('--n-fft', type=int, default=2048, help='number of fft (argument for stft)')
    parser.add_argument('--win-length', type=int, default=2048, help='window length (argument for stft)')
    parser.add_argument('--hop-length', type=int, default=512, help='hop length (argument for stft)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='sample rate to resample input wav file')
    parser.add_argument('--n-frame-in-segment', type=int, default=15, help='number of frames of spectrogram of a 2D segment')
    parser.add_argument('--batch-size', type=int, default=64, help='number of segments in a batch')
    parser.add_argument('--resample', type=bool, default=True, help='to resample input or not')
    args = vars(parser.parse_args())
    evaluate(**args)