import numpy as np
import h5py


# for deep learning training phase, I need to choose the data from each day
def Getter(path, validation=False, test_ratio=0.15, constraint=None):
    '''120 files from folder /05, /06, /07, seperate to training -- 80, validation -- 20, testing -- 20'''
    f = h5py.File(path, "r")
    file_size= len([i for i in f])
    if validation==True:
        test_size = round(file_size * 0.15)
        val_size = round(file_size * 0.15)
        train_size = file_size - test_size - val_size

        indices = np.arange(1,file_size+1)
        train_indices = np.random.choice(indices, size=train_size, replace=False)
        indices = np.delete(indices,train_indices)
        val_indices = np.random.choice(indices, size=val_size, replace=False)
        test_indices = np.delete(indices, val_indices)
    else:
        test_size = round(file_size * test_ratio)
        train_size = file_size - test_size
        indices = np.arange(1, file_size + 1)
        train_indices = np.random.choice(indices, size=train_size, replace=False)
        test_indices = np.delete(indices, train_indices)
        test_indices = np.random.choice(test_indices, size=test_size, replace=False)

    # Training set
    X_container = []
    y_container = []
    for j in train_indices:
    	#check whether this day's data are empty
        if "input%d" % j in f:
            X = f["input%d" % j][:]
            y = f["output%d" % j][:]
            shuffle_index = np.random.choice(len(y), size=len(y), replace=False)
            #nb_data I'd like to choose in one day
            # if constraint is None:
            #     to_use_index = np.random.choice(len(y), size=len(y), replace=False)
            # elif len(y) < constraint:
            #     to_use_index = np.random.choice(len(y), size=len(y), replace=False)
            # else:
            #     to_use_index = np.random.choice(len(y), size=constraint, replace=False)

            X_container.append(X[shuffle_index])
            y_container.append(y[shuffle_index])
        else:
            continue
    # concatenate all the data and get ready to pass them to the CNN model.
    X_train = np.concatenate(X_container, axis=0)
    y_train = np.concatenate(y_container, axis=0)

    # Validation set
    if validation == True:
        X_container = []
        y_container = []
        for j in val_indices:
            # check whether this day's data are empty
            if "input%d" % j in f:
                X = f["input%d" % j][:]
                y = f["output%d" % j][:]
                shuffle_index = np.random.choice(len(y), size=len(y), replace=False)
                X_container.append(X[shuffle_index])
                y_container.append(y[shuffle_index])
            else:
                continue
        # concatenate all the data and get ready to pass them to the CNN model.
        X_val = np.concatenate(X_container, axis=0)
        y_val = np.concatenate(y_container, axis=0)

    # Test set
    X_container = []
    y_container = []
    for j in test_indices:
        # check whether this day's data are empty
        if "input%d" % j in f:
            X = f["input%d" % j][:]
            y = f["output%d" % j][:]
            shuffle_index = np.random.choice(len(y), size=len(y), replace=False)
            X_container.append(X[shuffle_index])
            y_container.append(y[shuffle_index])
        else:
            continue
    # concatenate all the data and get ready to pass them to the CNN model.
    X_test = np.concatenate(X_container, axis=0)
    y_test = np.concatenate(y_container, axis=0)

    # Number of entries for each set of data
    if validation == True:
        print("Training dataset size: %d\nValidation dataset size: %d\nTesting dataset size: %d\n" \
              % (X_train.shape[0],X_val.shape[0],X_test.shape[0]))
        return [X_train, y_train], [X_val, y_val], [X_test, y_test]
    else:
        print("Training dataset size: %d\nTesting dataset size: %d\n" \
              % (X_train.shape[0], X_test.shape[0]))
    return [X_train, y_train], [X_test, y_test]

# path = "/home/wjchen/Documents/rhyang/data05-07.h5"
# [train, test] = Getter(path)
'''
import sketch_GenBatch as gb
path = "/home/wjchen/Documents/rhyang/data05-07.h5"
[train,val,test] = gb.Getter(path)
train[1].shape,val[1].shape,test[1].shape
'''