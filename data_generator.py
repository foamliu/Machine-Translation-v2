# encoding=utf-8
import json

import numpy as np
from torch.utils.data import Dataset

from config import *

def encode_text(word_map, c):
    return [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))


class TranslationDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split):
        """
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'train', 'valid', 'test-a', 'test-b'}

        print('loading {} samples'.format(split))
        if split == 'train':
            samples_path = 'data/samples_train.json'
        else:
            samples_path = 'data/samples_valid.json'

        self.samples = json.load(open(samples_path, 'r'))
        self.dataset_size = len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        path = os.path.join(self.image_folder, sample['image_id'])
        # Read images
        img = imread(path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 256, 256)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)

        # Sample captions
        captions = sample['caption']
        # Sanity check
        assert len(captions) == captions_per_image
        c = captions[i % captions_per_image]
        c = list(jieba.cut(c))
        # Encode captions
        enc_c = encode_caption(self.word_map, c)

        caption = torch.LongTensor(enc_c)

        caplen = torch.LongTensor([len(c) + 2])

        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor([encode_caption(self.word_map, jieba.cut(c)) for c in captions])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
