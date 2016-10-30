from sklearn.svm import SVR
import numpy as np
import h5py
from GenBatch import Getter
from Evaluation import R_square
import Constant as C

#working in progress: a svm model.
X_training, y_training = Getter("/home/rhyang/Documents/data.h5", 200, constraint=None)
X_eval, y_eval = Getter("/home/rhyang/Documents/data_val.h5", 100, constraint=None)
#################################################################################################
model = SVR(C=1, max_iter=100)
#randomly choose data from the file
X_training, y_training = None, None
index = np.random.choice(np.arange(len(f)/2)+1, size=C.nb_choice, replace=False)
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
X_training = np.concatenate(X_container, axis=0)
#normalize
sh = X_training.shape
prod = 1
for i in range(1, len(sh)):
	prod *= sh[i]
X_training = X_training.reshape((sh[0], prod))
X_training -= X_training.mean(axis=0)
X_training /= np.std(X_training, axis=0)
y_training = np.concatenate(y_container, axis=0)
model.fit(X_training, y_training)
X_eval -= X_eval.mean(axis=0)
X_eval /= np.std(X_eval, axis=0)
result = model.predict(X_eval)
print(result.shape, y_eval.shape)
print(R_square(y_eval, result))