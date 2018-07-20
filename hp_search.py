from __future__ import print_function

import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input, CuDNNLSTM, Bidirectional, TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.models import Model

from config import batch_size, num_train_samples, num_valid_samples, hidden_size, vocab_size_zh, embedding_size, Tx, Ty
from data_generator import DataGenSequence


def data():
    return DataGenSequence('train'), DataGenSequence('valid')


def create_model():
    input_tensor = Input(shape=(Tx, embedding_size), dtype='float32')
    x = Bidirectional(CuDNNLSTM({{choice([256, 512, 1024])}}, return_sequences=True))(input_tensor)
    x = Dropout({{uniform(0, 1)}})(x)
    x = CuDNNLSTM({{choice([256, 512, 1024])}}, return_sequences=True)(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = CuDNNLSTM({{choice([256, 512, 1024])}}, return_sequences=True)(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = TimeDistributed(Dense(vocab_size_zh, activation='softmax'))(x)
    output = x
    model = Model(inputs=input_tensor, outputs=output)

    decoder_target = tf.placeholder(dtype='float32', shape=(None, Ty))
    model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd', 'nadam'])}}, target_tensors=[decoder_target])

    model.fit_generator(
        DataGenSequence('train'),
        steps_per_epoch=num_train_samples / batch_size // 50,
        validation_data=DataGenSequence('valid'),
        validation_steps=num_valid_samples / batch_size // 50)

    score, acc = model.evaluate_generator(DataGenSequence('valid'), verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(DataGenSequence('valid')))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
