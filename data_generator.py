# encoding=utf-8
import json

from torch.utils.data import Dataset

from config import *


class TranslationDataset(Dataset):
    def __init__(self, split):
        self.split = split
        assert self.split in {'train', 'valid'}

        print('loading {} samples'.format(split))
        if split == 'train':
            samples_path = 'data/samples_train.json'
        else:
            samples_path = 'data/samples_valid.json'

        self.samples = json.load(open(samples_path, 'r'))
        self.dataset_size = len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        input_tensor = torch.tensor(sample['input'], dtype=torch.long, device=device).view(-1, 1)
        target_tensor = torch.tensor(sample['output'], dtype=torch.long, device=device).view(-1, 1)
        return (input_tensor, target_tensor)

    def __len__(self):
        return self.dataset_size
