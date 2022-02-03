import torch
from pathlib import Path

def is_extension_supported(f: Path):
    return '.wav' in f.suffixes or '.mp3' in f.suffixes

def get_device() -> torch.device:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

def load_model(model_path: str) -> torch.nn.Module:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Not a correct model path')
    model = torch.load(model_path)
    return model

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
