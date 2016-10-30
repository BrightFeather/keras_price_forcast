import csv, os, pickle
import numpy as np

#Pick mainstream contrast and save as a list
def is_liter(n):
    return not n.isdigit()


def is_digit(n):
    return n.isdigit()


def Getter(path, category):
    csvfile = open(path, 'r')
    stats = csv.DictReader(csvfile)
    column_list = []
    for column in stats:
        row_list = []
        row_list.append(column["InstrumentID"])
        row_list.append(column["Volume"])
        row_list.append(column["Date"])
        column_list.append(row_list)
    table = np.asarray(column_list)
    dic = {}
    max_num = 0
    for i in range(table.shape[0]):
        name_filter = "".join(list(filter(is_liter, table[i, 0])))
        number_filter = "".join(list(filter(is_digit, table[i, 0])))
        if (name_filter == category) and (int(table[i, 1]) > max_num):
            dic[category] = number_filter + "_" + table[i, 2]
            max_num = int(table[i, 1])

    return category + dic[category] + ".csv"


def Indexer(path):
    name_list = []
    for PATH in path:
        for root, dirs, files in os.walk(PATH):
            for name in files:
                if "stats" in name:
                    print(name)
                    name_list.append(Getter(root + "/" + name, "T"))
    return name_list

name_list = Indexer(["/home/wjchen/tick/CFFEX/stats/2016/07"])
f = open("/home/wjchen/Documents/Project/index07.txt", "wb")
pickle.dump(name_list, f)
f.close()