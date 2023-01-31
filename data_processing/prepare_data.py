import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from data_processing.transforms import train_transform, val_transform, test_transform

def get_class_encoding(root_path): 
    print(root_path)
    csv_file = pd.read_csv(root_path / 'train.csv')
    leaves_labels = sorted(list(set(csv_file['label'])))
    class_encoding = {k : v for v, k in enumerate(leaves_labels)}
    return class_encoding


def load_csv(root_path, mode):
    csv_file = pd.read_csv(root_path / f"{mode}.csv")
    return csv_file

class CustomDataset(Dataset):
    def __init__(self, args, mode, transform):
        self.root_path = args.data_path
        self.class_encoding = get_class_encoding(self.root_path)
        self.data_file = load_csv(self.root_path, mode)
        self.transform = transform
        self.mode = mode

        self.images = np.asarray(self.data_file['image'])
        if self.mode != 'test':
            self.labels = np.asarray(self.data_file['label'])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_file = Image.open(self.root_path / self.images[idx])
        target = self.labels[idx] if self.mode != 'test' else None

        return self.transform(image_file), self.class_encoding[target]


def create_dataloader(args, mode = 'train'):
    transform = train_transform if mode == 'train' else val_transform if mode == 'val' else test_transform
    leaf_dataset = CustomDataset(args, mode, transform)
    shuffle = True if mode == 'train' else False
    dataloader = DataLoader(dataset = leaf_dataset, batch_size=args.batch_size, shuffle=shuffle)

    return dataloader