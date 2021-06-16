import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        row = self.data[self.data.columns[0]][idx]
        label = self.data[self.data.columns[1]][idx]

        return row, label