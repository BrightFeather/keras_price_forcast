import csv, os, h5py, pickle
import numpy as np
import pandas as pd

# input_size = 100
# interval_length = 4
# sample rate of datasheet is around 2Hz

def FilePriceToData(index_path, path, data_path, input_size=100, interval_length=4, sample_rate= 5, constraint=50):
    nb_file = 0
    f = h5py.File(data_path, "w")
    # pick the mainstream file name list from the file
    name_list = []
    for index in index_path:
        index_file = open(index, "rb")
        # the content of the list: name of the file to use
        name_list += pickle.load(index_file)

    # Read data from tick
    for PATH in path:
        for root, dirs, files in os.walk(PATH):
            for name in files:
                if name in name_list:
                    # print(name)
                    csvfile = open(root + "/" + name, 'r')
                    df = pd.read_csv(csvfile)

                    # filter data: drop na columns, drop forms with negative value and forms with #row smaller than constraint
                    df = df.dropna()
                    if sum(df.LastPrice<0) > 0 or len(df) < constraint:
                        continue

                    nb_file += 1

                    price_column = list(df['LastPrice'])
                    mean_price = df['LastPrice'].mean()
                    downsampled_price_column = price_column[1::sample_rate] # Starting from the first one, downsample rate: 1/5
                    entry_size = len(downsampled_price_column)
                    inputPrice_list = []
                    targetPrice_list = []
                    nb_trainset = entry_size-input_size-interval_length
                    for i in range(nb_trainset):
                        # each training iteration takes input size of input_size
                        # *******************|--------------|+
                        # _____input data____|___interval___|target
                        inputPrice_list.append(downsampled_price_column[i:i+input_size])
                        targetPrice_list.append(downsampled_price_column[i+input_size+interval_length])
                    # use numpy array, subtract mean from price data
                    inputPrice_data = np.asarray(inputPrice_list).astype('float') - mean_price
                    targetPrice_data = np.asarray(targetPrice_list).astype('float') - mean_price


                    # max_value = target_set.max()
                    # mask2 = np.where(np.abs(target_set - max_value) > 20)[0]
                    #
                    # if len(mask1) > 0 and len(mask2) > 0:
                    #     mask = np.concatenate((mask1, mask2))
                    # elif len(mask1) > 0 and len(mask2) == 0:
                    #     mask = mask1
                    # elif len(mask1) == 0 and len(mask2) > 0:
                    #     mask = mask2
                    # if len(mask) > 0:
                    #     print(len(mask))
                    #     mask = np.unique(mask)
                    #     train_set = np.delete(train_set, mask, axis=0)
                    #     target_set = np.delete(target_set, mask)

                    # checkpoint(filter the whole day's data if all the data are BAD data(deleted))
                    # if train_set.shape[0] == 0:
                    #     continue

                    # target_set: return; shape(nb_samples,)
                    f.create_dataset("output%d" % nb_file,
                                     data=targetPrice_data, dtype="float")

                    # if (train_set < 0).any():
                    #     raise Warning("Bad data occur!")

                    # training_set: shape(nb_samples, time_to_look_back, nb_feature in each tick)
                    f.create_dataset("input%d" % nb_file,
                                     data=inputPrice_data, dtype="float")
                    # print(train_set.shape)
                    print(str(nb_trainset)+' entries added from file ' + str(name))

    f.close()
    print("number of files processed: " + str(nb_file))