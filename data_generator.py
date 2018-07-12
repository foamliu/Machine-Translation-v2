# encoding=utf-8
import pickle

import keras
import numpy as np
from gensim.models import KeyedVectors
from keras.utils import Sequence

from config import batch_size, vocab_size_zh, max_token_length_en, max_token_length_zh, unknown_word, unknown_embedding, \
    embedding_size


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        print('loading fasttext en word embedding')
        self.word_vectors_en = KeyedVectors.load_word2vec_format('data/wiki.en.vec')
        print('loading zh word embedding')
        self.word_vectors_zh = KeyedVectors.load_word2vec_format('data/sgns.merge.char')

        vocab_zh = pickle.load(open('data/vocab_train_zh.p', 'rb'))
        self.idx2word_zh = sorted(vocab_zh)
        self.word2idx_zh = dict(zip(self.idx2word_zh, range(len(vocab_zh))))

        vocab_en = pickle.load(open('data/vocab_train_en.p', 'rb'))
        self.idx2word_en = sorted(vocab_en)
        self.word2idx_en = dict(zip(self.idx2word_en, range(len(vocab_en))))

        if usage == 'train':
            samples_path = 'data/samples_train.p'
        else:
            samples_path = 'data/samples_valid.p'

        samples = pickle.load(open(samples_path, 'rb'))
        self.samples = samples
        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_y = np.empty((length, vocab_size_zh), dtype=np.int32)

        batch_text_embedding_en = np.zeros((length, max_token_length_en, embedding_size), np.float32)
        batch_text_embedding_zh = np.zeros((length, max_token_length_zh, embedding_size), np.float32)

        for i_batch in range(length):
            sample = self.samples[i + i_batch]

            for idx, word in enumerate(['input_en']):
                if word == unknown_word:
                    batch_text_embedding_en[i_batch, idx] = unknown_embedding
                else:
                    batch_text_embedding_en[i_batch, idx] = self.word_vectors_en[word]

            for idx, word in enumerate(sample['input_zh']):
                if word == unknown_word:
                    batch_text_embedding_zh[i_batch, idx] = unknown_embedding
                else:
                    batch_text_embedding_zh[i_batch, idx] = self.word_vectors_zh[word]

            batch_y[i_batch] = keras.utils.to_categorical(sample['output'], vocab_size_zh)

        return [batch_text_embedding_en, batch_text_embedding_zh], batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
