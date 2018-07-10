import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Embedding, TimeDistributed, Concatenate, Bidirectional
from keras.models import Model
from keras.utils import plot_model

from config import hidden_size, vocab_size_en, vocab_size_zh, embedding_size, max_token_length_en, max_token_length_zh


def build_model():
    en_input = Input(shape=(max_token_length_en,), dtype='int32')
    x = Embedding(input_dim=vocab_size_en, output_dim=embedding_size)(en_input)
    x = LSTM(hidden_size, return_sequence=False)(x)
    en_embedding = Dense(embedding_size)(x)

    zh_input = Input(shape=(max_token_length_zh,), dtype='int32')
    x = Embedding(input_dim=vocab_size_zh, output_dim=embedding_size)(zh_input)
    x = LSTM(hidden_size, return_sequence=True)(x)
    zh_embedding = TimeDistributed(Dense(embedding_size))(x)

    x = [en_embedding, zh_embedding]
    x = Concatenate(axis=1)(x)
    x = Bidirectional(LSTM(hidden_size, return_sequences=False))(x)

    output = Dense(vocab_size_zh, activation='softmax', name='output')(x)

    inputs = [en_input, zh_input]
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
