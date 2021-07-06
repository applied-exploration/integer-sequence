import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class CSVDataset(Dataset):
    def __init__(self, file_name, stringify = False):
        self.data = pd.read_csv(file_name)
        self.stringify = stringify

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        value = self.data[self.data.columns[0]][idx]
        label = self.data[self.data.columns[1]][idx]

        if self.stringify: value = ''.join(str(x)+',' for x in str(value))[:-1]

        return value, label