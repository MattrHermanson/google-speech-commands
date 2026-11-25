import numpy as np
import torch
from torch.utils.data import Dataset


class AudioClipDataset(Dataset):
    def __init__(self, data_path: str, labels_path: str, transform=None):

        # Layout: each `.X` line is one clip: 16000 integers, the raw 16-bit PCM waveform
        # samples (range -32768 to 32767) at 16000 Hz, mono. Clips shorter than 1 s are
        # zero-padded at the end.

        raw_data = np.loadtxt(data_path, dtype=np.float32, delimiter=" ")

        self.min = -32768.0
        self.max = -32768.0

        # Min-max scale to; add epsilon to prevent division by zero on silent samples
        data_norm = (raw_data - self.min) / (self.max - self.min + 1e-8)

        self.data = torch.from_numpy(data_norm)
        self.labels = torch.from_numpy(np.loadtxt(labels_path, dtype=np.int64, delimiter=" "))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    # TODO: implement this to change num to str
    # will need to get classes from classes.txt in __init__
    def label_to_name(self, label: int) -> str:
        return str(label)
