from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Activation, Dropout, LSTM, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from FileReader import *
from Evaluation import *
from GenBatch import Getter
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from scipy.stats import mode
import numpy as np
import pickle
import Constant as C
import matplotlib.pyplot as plt
import GenBatch_wj as gb
import time
sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)


def CNNconstruction():
    model = Sequential()
    model.add(Convolution2D(16, 1, 3, input_shape=(22, 1, C.nb_data-1), init='he_normal'))
    model.add(Activation("relu"))

    # model.add(MaxPooling2D((1, 2)))

    model.add(Convolution2D(32, 1, 3, init='he_normal'))
    model.add(Activation("relu"))

    # model.add(MaxPooling2D((1, 2)))

    model.add(Convolution2D(64, 1, 3, init='he_normal'))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((1,2)))

    model.add(Flatten())
    model.add(Dense(128, init="he_normal"))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, init="normal"))
    model.compile(optimizer=Adam(), loss='mse')
    return model


def LSTMconstruction():
    layer = [1,50,100,1]
    timesteps = 100
    batch_size = 64
    model = Sequential()
    model.add(LSTM(output_dim=layer[1], input_shape=(timesteps, 1), return_sequences=True)) #input_shape=(timesteps, 1) input_dim=1
    # model.add(LSTM(output_dim=layer[1], input_dim=1, return_sequences=True)) #input_shape = (timesteps,layer[0])
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=layer[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=layer[3], activation='linear'))
    model.compile(optimizer=RMSprop(lr=1e-4,rho=0.9,epsilon=1e-08), loss='mse')
    # print('Compilation time: ', time)
    print(model.summary())
    return model

def ESNconstruction():
    model = Sequential()
    # for i in range(1000):
        # model.add(ESN)

    model.add()
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08,decay=0)
    model.compile(optimizer=rmsprop)

def Complex(dataset):
    part1 = np.diff(dataset[:, :, :11], axis=1)
    part2 = dataset[:, :-1, 11:-1] / dataset[:, 1:, 11:-1]
    part3 = dataset[:, :-1, -1] - dataset[:, 1:, -1]
    part3 = np.expand_dims(part3, axis=2)
    part3 = np.tanh(part3)
    return np.concatenate((part1, part2, part3), axis=2)

# four methods to transfer or normalize data: details can be found in numpy module
def Diff(dataset):
    return np.diff(dataset, axis=1)


def Flat(dataset):
    return dataset.reshape(dataset.shape[0], np.prod(dataset.shape[1:]))


def Descale(dataset):
    dataset[:, :, :11] -= 100
    return dataset


def Zscore(dataset):
    cache = {}
    mean = dataset.mean(axis=0)
    dataset -= mean
    std = np.std(dataset, axis=0)
    dataset /= std
    cache["mean"] = mean
    cache["std"] = std
    return dataset, cache

# def run_model():

########################################################################################################################################
#save the model as a configuration file

print("Start building model...")
start_time = time.time()
model = LSTMconstruction() #RNN
config = model.get_config()
conf_file = open("/home/wjchen/Documents/Project/model/LSTM_configuration.txt", "wb")
pickle.dump(config, conf_file)
conf_file.close()

# callbacks; to avoid overfitting
ES = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
MC = ModelCheckpoint("/home/wjchen/Documents/Project/model/weights.{epoch02d}-{val_loss:.2f}.h5", monitor='val_loss', save_best_only=True)
print("Model built, took %.2f s " %(time.time()-start_time))
#######################################################################################################################################
# load data
print("Start loading data...")
start_time = time.time()
path = "/home/wjchen/Documents/Project/data/data05-07.h5"
[train,test] = gb.Getter(path)
X_train, y_train = train[0], train[1]
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1))
# X_val, y_val = val[0], val[1]
X_test, y_test = test[0], test[1]
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
print('Data loaded, took %.2f s. ' % (time.time() - start_time))

# train model
print('Start training...')
model.fit(X_train, y_train, nb_epoch=1, batch_size=64, validation_split=0.2, verbose=1, callbacks=[ES, MC], show_accuracy=True)
print('Training finished')
model.save_weights("/home/wjchen/Documents/Project/model/weights/wj_weights.h5", overwrite=True)

# run on test set
evaluation_result = model.predict(X_test)
plt.figure(0)
plt.hist(evaluation_result.flatten(), bins=1000, histtype='stepfilled')
plt.figure(1)
plt.hist(y_test, bins=1000, histtype='stepfilled')

# X_training, y_training = Getter("/home/wjchen/Documents/rhyang/data.h5", 200, constraint=None)
# X_training = Diff(X_training)
# X_training = X_training.transpose(0,2,1)
# X_training = np.expand_dims(X_training, axis=2)
#
# X_eval, y_eval = Getter("/home/wjchen/Documents/rhyang/data_val.h5", 100, constraint=None)
# X_eval = Diff(X_eval)
# X_eval = X_eval.transpose(0,2,1)
# X_eval = np.expand_dims(X_eval, axis=2)
#
# R2 = 0

# for i in range(C.nb_epoch):
#     print("EPOCH: " + str(i + 1) + "/" + str(C.nb_epoch))
#     model.fit(X_train, y_train.reshape(-1, 1), nb_epoch=1, batch_size=C.batch_size, verbose=2, callbacks=[])
#
#     evaluation_result = model.predict(X_eval)
#     R2_temp = R_square(y_eval, evaluation_result.flatten())
#     print("Out-sample R2: " + str(R2_temp))
#     print("mean: " + str(evaluation_result.flatten().mean()))
#     print("std: " + str(np.std(evaluation_result.flatten())))
#
#     if R2_temp > R2:
#         model.save_weights("/home/wjchen/Documents/rhyang/best_weights.h5", overwrite=True)
#         R2 = R2_temp
#     model.save_weights("/home/wjchen/Documents/rhyang/weights.h5", overwrite=True)
#
# plt.figure(0)
# plt.hist(evaluation_result.flatten(), bins=1000, histtype='stepfilled')
# plt.figure(1)
# plt.hist(y_eval, bins=1000, histtype='stepfilled')

#REMARK: More details about Keras can be found at: https://keras.io;
#        More details about Theano can be founf at: https://www.deeplearning.net/software/theano/