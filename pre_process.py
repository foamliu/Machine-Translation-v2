import json
import xml.etree.ElementTree
from collections import Counter

import jieba
import nltk
from tqdm import tqdm

from config import *


def encode_text(word_map, c):
    return [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c) - 2)


def build_wordmap_zh():
    translation_path = os.path.join(train_translation_folder, train_translation_zh_filename)

    with open(translation_path, 'r') as f:
        sentences = f.readlines()

    word_freq = Counter()

    for sentence in tqdm(sentences):
        seg_list = jieba.cut(sentence)
        # Update word frequency
        word_freq.update(seg_list)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<start>'] = 0
    word_map['<end>'] = 1
    word_map['<unk>'] = 2
    word_map['<pad>'] = 3
    print(len(word_map))
    print(words[:10])

    with open('data/WORDMAP_zh.json', 'w') as file:
        json.dump(word_map, file, indent=4)


def build_wordmap_en():
    translation_path = os.path.join(train_translation_folder, train_translation_en_filename)

    with open(translation_path, 'r') as f:
        sentences = f.readlines()

    word_freq = Counter()

    for sentence in tqdm(sentences):
        sentence_en = sentence.strip().lower()
        tokens = nltk.word_tokenize(sentence_en)
        # Update word frequency
        word_freq.update(tokens)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<start>'] = 0
    word_map['<end>'] = 1
    word_map['<unk>'] = 2
    word_map['<pad>'] = 3
    print(len(word_map))
    print(words[:10])

    with open('data/WORDMAP_en.json', 'w') as file:
        json.dump(word_map, file, indent=4)


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
        out_file.write('\n'.join(data_en) + '\n')

    root = xml.etree.ElementTree.parse(os.path.join(valid_translation_folder, 'valid.en-zh.zh.sgm')).getroot()
    data_zh = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_translation_folder, 'valid.zh'), 'w') as out_file:
        out_file.write('\n'.join(data_zh) + '\n')


def build_samples():
    word_map_zh = json.load(open('data/WORDMAP_zh.json', 'r'))
    word_map_en = json.load(open('data/WORDMAP_en.json', 'r'))

    for usage in ['train', 'valid']:
        if usage == 'train':
            translation_path_en = os.path.join(train_translation_folder, train_translation_en_filename)
            translation_path_zh = os.path.join(train_translation_folder, train_translation_zh_filename)
            filename = 'data/samples_train.json'
        else:
            translation_path_en = os.path.join(valid_translation_folder, valid_translation_en_filename)
            translation_path_zh = os.path.join(valid_translation_folder, valid_translation_zh_filename)
            filename = 'data/samples_valid.json'

        print('loading {} texts and vocab'.format(usage))
        with open(translation_path_en, 'r') as f:
            data_en = f.readlines()

        with open(translation_path_zh, 'r') as f:
            data_zh = f.readlines()

        print('building {} samples'.format(usage))
        samples = []
        for idx in tqdm(range(len(data_en))):
            sentence_en = data_en[idx].strip().lower()
            tokens = nltk.word_tokenize(sentence_en)
            input_en = encode_text(word_map_en, tokens)

            sentence_zh = data_zh[idx].strip()
            seg_list = jieba.cut(sentence_zh)
            output_zh = encode_text(word_map_zh, list(seg_list))

            if len(input_en) <= max_len + 2 and len(output_zh) <= max_len + 2:
                samples.append({'input': list(input_en), 'output': list(output_zh)})

        with open(filename, 'w') as f:
            json.dump(samples, f)
        print('{} {} samples created at: {}.'.format(len(samples), usage, filename))


if __name__ == '__main__':
    build_wordmap_zh()
    build_wordmap_en()
    extract_valid_data()
    build_samples()
