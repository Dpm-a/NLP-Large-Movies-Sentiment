import torch
import pandas as pd
from torch.utils.data import Dataset


# Creating torch dataset
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

        if labels is not None:
            assert len(self.encodings['input_ids']) == len(
                self.labels), "Mismatch in lengths of encodings and labels"

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to("mps")
                for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx]).to("mps")
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
