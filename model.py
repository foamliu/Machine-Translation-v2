import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, TimeDistributed, Concatenate, Bidirectional, RepeatVector
from keras.models import Model
from keras.utils import plot_model

from config import hidden_size, vocab_size_zh, embedding_size, max_token_length_en, max_token_length_zh


def build_model():
    en_input = Input(shape=(max_token_length_en, embedding_size), dtype='float32')
    x = LSTM(hidden_size, return_sequences=False)(en_input)
    x = Dense(embedding_size)(x)
    en_embedding = RepeatVector(1)(x)

    zh_input = Input(shape=(max_token_length_zh, embedding_size), dtype='float32')
    x = LSTM(hidden_size, return_sequences=True)(zh_input)
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
