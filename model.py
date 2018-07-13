import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Concatenate, Bidirectional, RepeatVector, Activation, Dot
from keras.models import Model
from keras.utils import plot_model

from config import n_s, n_a, vocab_size_zh, embedding_size, Tx, Ty


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def one_step_attention(a, s_prev):
    print('a.shape: ' + str(a.shape))
    s_prev = RepeatVector(Tx)(s_prev)
    concat = Concatenate(axis=-1)([a, s_prev])
    print('concat.shape: ' + str(concat.shape))
    e = Dense(10, activation="tanh")(concat)
    energies = Dense(1, activation="relu")(e)
    alphas = Activation(softmax)(energies)
    print('alphas.shape: ' + str(alphas.shape))
    context = Dot(axes=1)([alphas, a])
    print('context.shape: ' + str(context.shape))
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
