import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Concatenate, Bidirectional, RepeatVector, Activation, \
    Dot
from keras.models import Model
from keras.utils import plot_model

from config import n_s, n_a, vocab_size_zh, embedding_size, Tx, Ty


def one_step_attention(a, s_prev):
    s_prev = RepeatVector(Tx)(s_prev)
    concat = Concatenate(axis=-1)([a, s_prev])
    e = Dense(10, activation="tanh")(concat)
    energies = Dense(1, activation="relu")(e)
    alphas = Activation('softmax')(energies)
    context = Dot(axes=1)([alphas, a])
    return context


def build_model():
    X = Input(shape=(Tx, embedding_size), dtype='float32')
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    print('a.shape: ' + str(a.shape))

    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = LSTM(n_s, return_state=True)(context, initial_state=[s, c])
        out = Dense(vocab_size_zh, activation='softmax')(s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
