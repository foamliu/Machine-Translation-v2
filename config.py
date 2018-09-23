import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 10000
patience = 50
num_train_samples = 8852422
num_valid_samples = 7613
embedding_size = 300
vocab_size_zh = 50000
max_token_length_en = Tx = 20 + 1  # 1 is for tailing stop word
max_token_length_zh = Ty = 20 + 1  # 1 is for tailing stop word
MAX_LENGTH = 21
teacher_forcing_ratio = 0.5
hidden_size = 256
min_word_freq = 3
max_len = 20

train_folder = 'data/ai_challenger_translation_train_20170912'
valid_folder = 'data/ai_challenger_translation_validation_20170912'
test_a_folder = 'data/ai_challenger_translation_test_a_20170923'
test_b_folder = 'data/ai_challenger_translation_test_b_20171128'
train_translation_folder = os.path.join(train_folder, 'translation_train_20170912')
valid_translation_folder = os.path.join(valid_folder, 'translation_validation_20170912')
train_translation_en_filename = 'train.en'
train_translation_zh_filename = 'train.zh'
valid_translation_en_filename = 'valid.en'
valid_translation_zh_filename = 'valid.zh'

SOS_token = 0
EOS_token = 1

start_word = '<start>'
stop_word = '<end>'
unknown_word = '<unk>'
