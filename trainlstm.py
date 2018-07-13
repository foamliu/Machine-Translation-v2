from keras.layers import *
from keras.models import *
import multi_gpu_utils2 as multi_gpu_utils
from keras.layers import *
from keras.models import *

import multi_gpu_utils2 as multi_gpu_utils


def DNN(featurec, outputc):
    input = Input(shape=featurec)
    x1 = Dense(outputc, activation='linear')(input)
    return Model(input, x1)


# single LSTM layer
def test_lstm_stateful_withGPU_case1():
    gpusize = 8
    lstmbatch_size = gpusize * 4
    sample_size = lstmbatch_size * 20
    timestep = 20
    testdata = np.random.rand(lstmbatch_size, timestep, 1)
    inputs = Input(batch_shape=(lstmbatch_size, None, 1))
    lstm = LSTM(1, stateful=True)
    m1 = Model(inputs, lstm(inputs))
    print("predict with single GPU: ")
    print(m1.predict(testdata))
    # m2 = multi_gpu_model(m1)
    # m2.predict(testdata, gpus=gpusize)#<<--- invalid state size in origin version of multi_gpu_model
    m1.reset_states()
    print("predict with multiple GPUs: ")
    m3 = multi_gpu_utils.multi_gpu_model(m1, gpus=gpusize)
    print(m3.predict(testdata))
    m3.compile(loss='mean_squared_error', optimizer='sgd')
    datax = np.random.rand(sample_size, timestep, 1)
    datay = np.random.rand(sample_size, 1)
    print("training with multiple GPUs: ")
    m3.fit(datax, datay, epochs=2, batch_size=lstmbatch_size)


# DNN=>LSTMs=>DNN
def test_lstm_stateful_withGPU_case2():
    gpusize = 8
    lstmbatch_size = gpusize * 4
    sample_size = lstmbatch_size * 20
    timestep = 2
    features_size = [(1,), (2,), (2,), (1,)]
    state_feature_size = features_size[2][0]
    testdata = np.random.rand(lstmbatch_size, timestep, 1)
    inputs = Input(batch_shape=(lstmbatch_size, None, 1))
    lstm1 = LSTM(state_feature_size, stateful=True, return_sequences=True)
    lstm2 = LSTM(state_feature_size, stateful=True, return_sequences=True)
    dnnencode = DNN(features_size[0], features_size[1][0])
    dnndecode = DNN(features_size[2], features_size[3][0])
    dnnencodes = TimeDistributed(dnnencode, input_shape=features_size[0])
    dnndecodes = TimeDistributed(dnndecode, input_shape=features_size[2])
    # DNN1=>LSTM1=>LSTM2=>DNN2
    output = dnndecodes(lstm2(lstm1(dnnencodes(inputs))))
    m1 = Model(inputs, output)
    print("predict with single GPU: ")
    print(m1.predict(testdata))
    m1.reset_states()
    print("predict with multiple GPUs: ")
    m3 = multi_gpu_utils.multi_gpu_model(m1, gpus=gpusize)
    print(m3.predict(testdata))
    m3.compile(loss='mean_squared_error', optimizer='sgd')
    datax = np.random.rand(sample_size, timestep, 1)
    datay = np.random.rand(sample_size, timestep, 1)
    print("training with multiple GPUs: ")
    m3.fit(datax, datay, epochs=2, batch_size=lstmbatch_size)


# single LSTM2D layer
def test_lstm_stateful_withGPU_case3():
    gpusize = 8
    lstmbatch_size = gpusize * 4
    sample_size = lstmbatch_size * 20
    timestep = 3
    imgsize = (8, 8, 3)
    testdata = np.random.rand(*(lstmbatch_size, timestep) + imgsize)
    inputshape = (lstmbatch_size, None) + imgsize
    inputs = Input(batch_shape=inputshape)
    lstm2d = ConvLSTM2D(filters=3, kernel_size=(3, 3), padding="same", return_sequences=False, stateful=False,
                        use_bias=True)
    m1 = Model(inputs, lstm2d(inputs))
    print("predict with single GPU: ")
    print(m1.predict(testdata))
    m1.reset_states()
    m3 = multi_gpu_utils.multi_gpu_model(m1, gpus=gpusize)
    print("predict with multiple GPUs: ")
    print(m3.predict(testdata))
    m3.compile(loss='mean_squared_error', optimizer='sgd')
    datax = np.random.rand(*((sample_size, timestep) + imgsize))
    datay = np.random.rand(*((sample_size,) + imgsize))
    print("training with multiple GPUs: ")
    m3.fit(datax, datay, epochs=2, batch_size=lstmbatch_size)


if __name__ == "__main__":
    test_lstm_stateful_withGPU_case1()
    test_lstm_stateful_withGPU_case2()
    test_lstm_stateful_withGPU_case3()
