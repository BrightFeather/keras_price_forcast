from keras.models import Sequential
from FileReader import *
from Evaluation import *
from scipy.stats import linregress
import h5py, pickle
import numpy as np
import Constant as C
import matplotlib.pyplot as plt


conf_file = open("/home/rhyang/Documents/CNN_configuration_alter2.txt", "rb")
config = pickle.load(conf_file)
model = Sequential.from_config(config)
model.load_weights("/home/rhyang/Documents/weights_alter2.h5")

FileToData("/home/rhyang/Documents/tick/CFFEX/IC/day/2016/07",
           "/home/rhyang/Documents/data_val.h5", C.nb_data, C.interval, None)
f = h5py.File("/home/rhyang/Documents/data_val.h5", "r")

index = np.random.choice(np.arange(len(f))+1, size=C.nb_choice_predict, replace=False)
X_container = []
y_container = []
for j in index:
    if "training%d" % j in f:
        X = f["training%d" % j][:]
        y = f["delta%d" % j][:]
        to_use_index = np.random.choice(len(y), size=len(y), replace=False)
        X_container.append(X[to_use_index])
        y_container.append(y[to_use_index])
    else:
        continue
X_eval = np.concatenate(X_container, axis=0)
y_eval = np.concatenate(y_container, axis=0)
X_eval_nor = X_eval - X_eval.mean(axis=0)
X_eval_nor /= np.std(X_eval_nor, axis=0)

y_hat = model.predict(X_eval_nor)

R2 = linregress(y_hat.flatten(), y_eval.flatten())[2]
print(R2)

