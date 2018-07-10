import json
import os
import pickle
import zipfile

import jieba
from tqdm import tqdm

from config import start_word, stop_word, unknown_word
from config import train_annotations_filename
from config import train_folder, valid_folder, test_a_folder, test_b_folder


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def build_train_vocab():
    annotations_path = os.path.join(train_folder, train_annotations_filename)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    print('building {} train vocab')
    vocab = set()
    for a in tqdm(annotations):
        caption = a['caption']
        for c in caption:
            seg_list = jieba.cut(c)
            for word in seg_list:
                vocab.add(word)

    vocab.add(start_word)
    vocab.add(stop_word)
    vocab.add(unknown_word)

    filename = 'data/vocab_train.p'
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(vocab, encoded_pickle)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    # if not os.path.isdir(train_image_folder):
    extract(train_folder)

    # if not os.path.isdir(valid_image_folder):
    extract(valid_folder)

    # if not os.path.isdir(test_a_image_folder):
    extract(test_a_folder)

    # if not os.path.isdir(test_b_image_folder):
    extract(test_b_folder)
