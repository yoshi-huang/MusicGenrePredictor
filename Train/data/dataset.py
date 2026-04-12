import numpy as np
import torch
from torch.utils.data import Dataset

class MelDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.data = torch.tensor(np.load(x_path)).float()
        self.labels = torch.tensor(np.load(y_path)).float()
        self.attention_mask = (self.data != 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }
