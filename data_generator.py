# encoding=utf-8
import pickle

import keras
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.utils import Sequence

from config import batch_size, vocab_size_zh, max_token_length_en, max_token_length_zh, unknown_word, unknown_embedding


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
        text_embedding_en = []
        text_embedding_zh = []

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            input_en = []
            for word in sample['input_en']:
                if word == unknown_word:
                    input_en.append(unknown_embedding)
                else:
                    input_en.append(self.word_vectors_en[word])

            text_embedding_en.append(input_en)
            input_zh = []
            for word in sample['input_zh']:
                if word == unknown_word:
                    input_zh.append(unknown_embedding)
                else:
                    input_zh.append(self.word_vectors_zh[word])
            text_embedding_zh.append(input_zh)

            batch_y[i_batch] = keras.utils.to_categorical(sample['output'], vocab_size_zh)

        batch_text_embedding_en = sequence.pad_sequences(text_embedding_en, maxlen=max_token_length_en, padding='post')
        batch_text_embedding_zh = sequence.pad_sequences(text_embedding_zh, maxlen=max_token_length_zh, padding='post')
        return [batch_text_embedding_en, batch_text_embedding_zh], batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
