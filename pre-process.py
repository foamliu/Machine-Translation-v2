import os
import pickle
import zipfile
from collections import Counter
import xml.etree.ElementTree
import jieba
import nltk
from gensim.models import KeyedVectors
from tqdm import tqdm

from config import start_word, stop_word, unknown_word
from config import train_folder, valid_folder, test_a_folder, test_b_folder
from config import train_translation_folder, train_translation_zh_filename, train_translation_en_filename
from config import valid_translation_folder, valid_translation_zh_filename, valid_translation_en_filename


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def build_train_vocab_zh():
    print('loading zh word embedding')
    word_vectors = KeyedVectors.load_word2vec_format('data/sgns.merge.char')
    translation_path = os.path.join(train_translation_folder, train_translation_zh_filename)

    with open(translation_path, 'r') as f:
        data = f.readlines()

    vocab = []
    max_len = 0
    longest_sentence = None
    print('scanning train data (zh)')
    for sentence in tqdm(data):
        seg_list = jieba.cut(sentence.strip().lower())
        length = 0
        for word in seg_list:
            vocab.append(word)
            length = length + 1

        if length > max_len:
            longest_sentence = '/'.join(seg_list)
            max_len = length

    counter = Counter(vocab)
    total_count = 0
    covered_count = 0
    for word in tqdm(counter.keys()):
        total_count += counter[word]
        try:
            v = word_vectors[word]
            covered_count += counter[word]
        except (NameError, KeyError):
            #print(word)
            pass

    vocab = list(word_vectors.vocab.keys())
    vocab.append(start_word)
    vocab.append(stop_word)
    vocab.append(unknown_word)

    print('max_len(zh): ' + str(max_len))
    print('count of words in text (zh): ' + str(len(list(counter.keys()))))
    print('fasttext vocab size (zh): ' + str(len(vocab)))
    print('coverage: ' + str(covered_count / total_count))
    print('longest_sentence: ' + longest_sentence)

    filename = 'data/vocab_train_zh.p'
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(sorted(vocab), encoded_pickle)


def build_train_vocab_en():
    print('loading fasttext en word embedding')
    word_vectors = KeyedVectors.load_word2vec_format('data/wiki.en.vec')
    translation_path = os.path.join(train_translation_folder, train_translation_en_filename)

    with open(translation_path, 'r') as f:
        data = f.readlines()

    vocab = []
    max_len = 0
    longest_sentence = None
    print('building {} train vocab (en)')
    for sentence in tqdm(data):
        tokens = nltk.word_tokenize(sentence.strip().lower())
        for token in tokens:
            vocab.append(token)

        length = len(tokens)
        if length > max_len:
            longest_sentence = '/'.join(tokens)
            max_len = length

    counter = Counter(vocab)
    total_count = 0
    covered_count = 0
    for word in tqdm(counter.keys()):
        total_count += counter[word]
        try:
            v = word_vectors[word]
            covered_count += counter[word]
        except (NameError, KeyError):
            #print(word)
            pass

    vocab = list(word_vectors.vocab.keys())
    vocab.append(start_word)
    vocab.append(stop_word)
    vocab.append(unknown_word)

    print('max_len(zh): ' + str(max_len))
    print('count of words in text (en): ' + str(len(list(counter.keys()))))
    print('fasttext vocab size (en): ' + str(len(vocab)))
    print('coverage: ' + str(covered_count / total_count))
    print('longest_sentence: ' + longest_sentence)

    filename = 'data/vocab_train_en.p'
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(sorted(vocab), encoded_pickle)


def extract_valid_data():
    valid_translation_path = os.path.join(valid_translation_folder, 'valid.en-zh.en.sgm')
    with open(valid_translation_path, 'r') as f:
        data_en = f.readlines()
    data_en = [line.replace(' & ', ' &amp; ') for line in data_en]
    with open(valid_translation_path, 'w') as f:
        f.writelines(data_en)

    root = xml.etree.ElementTree.parse(valid_translation_path).getroot()
    data_en = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_translation_folder, 'valid.en'), 'w') as out_file:
        out_file.writelines(data_en)

    root = xml.etree.ElementTree.parse(os.path.join(valid_translation_folder, 'valid.en-zh.zh.sgm')).getroot()
    data_zh = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_translation_folder, 'valid.zh'), 'w') as out_file:
        out_file.writelines(data_zh)


def build_train_samples():
    print('loading fasttext en word embedding')
    word_vectors_en = KeyedVectors.load_word2vec_format('data/wiki.en.vec')
    translation_path_en = os.path.join(train_translation_folder, train_translation_en_filename)
    with open(translation_path_en, 'r') as f:
        data_en = f.readlines()
    vocab_en = pickle.load(open('data/vocab_train_en.p', 'rb'))
    idx2word_en = sorted(vocab_en)
    word2idx_en = dict(zip(idx2word_en, range(len(vocab_en))))

    print('loading zh word embedding')
    word_vectors_zh = KeyedVectors.load_word2vec_format('data/sgns.merge.char')
    translation_path_zh = os.path.join(train_translation_folder, train_translation_zh_filename)
    with open(translation_path_zh, 'r') as f:
        data_zh = f.readlines()
    vocab_zh = pickle.load(open('data/vocab_train_zh.p', 'rb'))
    idx2word_zh = sorted(vocab_zh)
    word2idx_zh = dict(zip(idx2word_zh, range(len(vocab_zh))))

    print('building train samples')
    samples = []
    for idx, sentence_en in tqdm(enumerate(data_en)):
        input_en = []
        tokens = nltk.word_tokenize(sentence_en.strip().lower())
        for word in tokens:
            try:
                v = word_vectors_en[word]
            except (NameError, KeyError):
                word = unknown_word
            input_en.append(word2idx_en[word])
        input_en.append(word2idx_en[stop_word])

        sentence_zh = data_zh[idx].strip().lower()
        seg_list = jieba.cut(sentence_zh)
        input_zh = []
        last_word = start_word
        for j, word in enumerate(seg_list):
            try:
                v = word_vectors_zh[word]
            except (NameError, KeyError):
                word = unknown_word

            input_zh.append(word2idx_zh[last_word])
            samples.append({'input_en': list(input_en), 'input_zh': list(input_zh), 'output': word2idx_zh[word]})
            last_word = word
        input_zh.append(word2idx_zh[last_word])
        samples.append({'input_en': list(input_en), 'input_zh': list(input_zh), 'output': word2idx_zh[stop_word]})

    filename = 'data/samples_train.p'
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)
    print('{} samples created.'.format(len(samples)))


def build_samples(usage):
    print('loading fasttext en word embedding')
    word_vectors_en = KeyedVectors.load_word2vec_format('data/wiki.en.vec')
    print('loading zh word embedding')
    word_vectors_zh = KeyedVectors.load_word2vec_format('data/sgns.merge.char')

    for usage in ['train', 'valid']:
        if usage == 'train':
            translation_path_en = os.path.join(train_translation_folder, train_translation_en_filename)
            translation_path_zh = os.path.join(train_translation_folder, train_translation_zh_filename)
            filename = 'data/samples_train.p'
        else:
            translation_path_en = os.path.join(valid_translation_folder, valid_translation_en_filename)
            translation_path_zh = os.path.join(valid_translation_folder, valid_translation_zh_filename)
            filename = 'data/samples_valid.p'

        print('loading {} texts and vocab'.format(usage))
        with open(translation_path_en, 'r') as f:
            data_en = f.readlines()
        vocab_en = pickle.load(open('data/vocab_train_en.p', 'rb'))
        idx2word_en = sorted(vocab_en)
        word2idx_en = dict(zip(idx2word_en, range(len(vocab_en))))

        with open(translation_path_zh, 'r') as f:
            data_zh = f.readlines()
        vocab_zh = pickle.load(open('data/vocab_train_zh.p', 'rb'))
        idx2word_zh = sorted(vocab_zh)
        word2idx_zh = dict(zip(idx2word_zh, range(len(vocab_zh))))

        print('building {} samples'.format(usage))
        samples = []
        for idx, sentence_en in tqdm(enumerate(data_en)):
            input_en = []
            tokens = nltk.word_tokenize(sentence_en.strip().lower())
            for word in tokens:
                try:
                    v = word_vectors_en[word]
                except (NameError, KeyError):
                    word = unknown_word
                input_en.append(word2idx_en[word])
            input_en.append(word2idx_en[stop_word])

            sentence_zh = data_zh[idx].strip().lower()
            seg_list = jieba.cut(sentence_zh)
            input_zh = []
            last_word = start_word
            for j, word in enumerate(seg_list):
                try:
                    v = word_vectors_zh[word]
                except (NameError, KeyError):
                    word = unknown_word

                input_zh.append(word2idx_zh[last_word])
                samples.append({'input_en': list(input_en), 'input_zh': list(input_zh), 'output': word2idx_zh[word]})
                last_word = word
            input_zh.append(word2idx_zh[last_word])
            samples.append({'input_en': list(input_en), 'input_zh': list(input_zh), 'output': word2idx_zh[stop_word]})

        with open(filename, 'wb') as f:
            pickle.dump(samples, f)
        print('{} {} samples created at: {}.'.format(len(samples), usage, filename))


if __name__ == '__main__':
    ensure_folder('data')

    if not os.path.isdir(train_folder):
        extract(train_folder)

    if not os.path.isdir(valid_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_folder):
        extract(test_a_folder)

    if not os.path.isdir(test_b_folder):
        extract(test_b_folder)

    if not os.path.isfile('data/vocab_train_zh.p'):
        build_train_vocab_zh()

    if not os.path.isfile('data/vocab_train_en.p'):
        build_train_vocab_en()

    extract_valid_data()

    if not os.path.isfile('data/samples_train.p') or not os.path.isfile('data/samples_valid.p'):
        build_samples()
