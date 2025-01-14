import torch
from torch.utils.data import Dataset, DataLoader
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
import numpy as np

class PPGDataset(Dataset):
    def __init__(self, data_dirs, config, transform=None):
        self.loader = UBFCrPPGLoader(
            name=config.DATASET.NAME,
            data_path=config.DATASET.ROOT,
            config_data=config
        )
        self.data_dirs = data_dirs
        self.transform = transform
        self.clip_length = config.DATASET.CLIP_LENGTH
        
    def __len__(self):
        return len(self.data_dirs)
        
    def __getitem__(self, idx):
        data_dir = self.data_dirs[idx]
        
        # Load video frames
        frames = self.loader.read_video(f"{data_dir['path']}/vid.avi")
        # Load ground truth PPG signal
        bvp = self.loader.read_wave(f"{data_dir['path']}/ground_truth.txt")
        
        # Create clips
        if len(frames) > self.clip_length:
            start_idx = np.random.randint(0, len(frames) - self.clip_length)
            frames = frames[start_idx:start_idx + self.clip_length]
            bvp = bvp[start_idx:start_idx + self.clip_length]
        
        # Convert to torch tensors
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2) / 255.0  # [C, T, H, W]
        bvp = torch.FloatTensor(bvp)
        
        return {'frames': frames, 'bvp': bvp}
