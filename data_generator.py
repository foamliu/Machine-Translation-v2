# encoding=utf-8
import json
import torch
from torch.utils.data import Dataset


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
        return sample['input'], sample['output']

    def __len__(self):
        return self.dataset_size
