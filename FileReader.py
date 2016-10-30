import csv, os, h5py, pickle
import numpy as np


def MidPrice(arr1, arr2):
    return (arr1 + arr2) / 2

#Generate data for Deep Learning Process
def FileToData(index_path, path, data_path, nb_data, interval, constraint=50):
    nb_file = 0
    f = h5py.File(data_path, "w")
    #pick the mainstream file name list from the file
    name_list = []
    for index in index_path:
        index_file = open(index, "rb")
        # the content of the list: name of the file to use
        name_list += pickle.load(index_file)

    #Read data from tick
    for PATH in path:
        for root, dirs, files in os.walk(PATH):
            for name in files:
                if name in name_list:
                    print(name)
                    nb_file += 1
                    csvfile = open(root + "/" + name, 'r')
                    datafile = csv.DictReader(csvfile)
                    column_list = []
                    # read csv files and save the data as a two dimentions array (nb_tick_in that day, the feature in each tick)
                    for row in datafile:
                        # for every row in datafile
                        row_list = []
                        for index in range(1, 6):
                            row_list.append(row['BidPrice%d' % index])
                            row_list.append(row['AskPrice%d' % index])
                        row_list.append(row['LastPrice'])
                        row_list.append(row['Volume'])
                        for index in range(1, 6):
                            row_list.append(row['BidVolume%d' % index])
                            row_list.append(row['AskVolume%d' % index])
                        # row_list.append(column['OpenInterest'])
                        column_list.append(row_list)

                    # use numpy array
                    day_data = np.asarray(column_list).astype('float')

                    # checkpoint(filter the file with a small amount of ticks in one day)
                    if len(day_data) < constraint:
                        continue

                    #training data generator with shape (nb_samples, time_to_look_back, nb_feature_in_each_tick)
                    N, _ = day_data.shape # no.row * no.col(22)
                    train_set_list = []
                    for i in range(N - (interval + nb_data)):
                        train_set_list.append(day_data[i:i + nb_data])
                    train_set = np.asarray(train_set_list)
                    del train_set_list

                    # print(train_set.shape)

                    # filter
                    # delete "BAD" data(data less than zero) from training and target set
                    mask = np.array([])
                    mask1 = np.where(train_set < 0.)[0] # negative values
                    target_set = MidPrice(day_data[nb_data + interval:, 0], # Mid price between bid and ask
                                          day_data[nb_data + interval:, 1])

                    max_value = target_set.max()
                    mask2 = np.where(np.abs(target_set - max_value) > 20)[0]

                    if len(mask1)>0 and len(mask2)>0:
                        mask = np.concatenate((mask1, mask2))
                    elif len(mask1)>0 and len(mask2)==0:
                        mask = mask1
                    elif len(mask1)==0 and len(mask2)>0:
                        mask = mask2
                    if len(mask)>0:
                        print(len(mask))
                        mask = np.unique(mask)
                        train_set = np.delete(train_set, mask, axis=0)
                        target_set = np.delete(target_set, mask)


                    # checkpoint(filter the whole day's data if all the data are BAD data(deleted))
                    if train_set.shape[0] == 0:
                        continue

                    #target_set: return; shape(nb_samples,)
                    f.create_dataset("delta%d" % nb_file,
                                     data=target_set - MidPrice(train_set[:, -1, 0], train_set[:, -1, 1]),
                                     dtype="float")

                    if (train_set < 0).any():
                        raise Warning("Bad data occur!")

                    #training_set: shape(nb_samples, time_to_look_back, nb_feature in each tick)
                    f.create_dataset("training%d" %
                                     nb_file, data=train_set, dtype="float")
                    print(train_set.shape)


    f.close()
    print("nb_file: " + str(nb_file))
#REMARK: I generate a dataset using h5py module; in this dataset, all the data are classified according to the day they belong to.
#Each day we have X:(nb_samples, time_to_look_back, nb_feature_in_a_tick), y(nb_samples, )

