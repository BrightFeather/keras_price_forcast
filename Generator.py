import Constant as C
import FileReader as fr
import FilePriceReader as fpr
# generate data.h5 and data_val.h5 files

# FileToData(index_path, path, data_path, nb_data, interval, constraint=50)
fr.FileToData(["/home/wjchen/Documents/Project/index05.txt",
               "/home/wjchen/Documents/Project/index06.txt"],
              ["/home/wjchen/tick/CFFEX/T/day/2016/05",
               "/home/wjchen/tick/CFFEX/T/day/2016/06"], "/home/wjchen/Documents/rhyang/data.h5", C.nb_data,
              C.interval)
fr.FileToData(["/home/wjchen/Documents/Project/index07.txt"], ["/home/wjchen/tick/CFFEX/T/day/2016/07"],
              "/home/wjchen/Documents/rhyang/data_val.h5", C.nb_data,
              C.interval)

# Added by Weijia. We only take the LastPrice in each form
fpr.FilePriceToData(["/home/wjchen/Documents/Project/index05.txt",
                     "/home/wjchen/Documents/Project/index06.txt"],
                    ["/home/wjchen/tick/CFFEX/T/day/2016/05",
                     "/home/wjchen/tick/CFFEX/T/day/2016/06"], "/home/wjchen/Documents/rhyang/dataxx.h5")

fpr.FilePriceToData(["/home/wjchen/Documents/Project/index07.txt"], ["/home/wjchen/tick/CFFEX/T/day/2016/07"],
              "/home/wjchen/Documents/rhyang/dataxx_val.h5")

fpr.FilePriceToData(["/home/wjchen/Documents/Project/index05.txt",
                     "/home/wjchen/Documents/Project/index06.txt",
                     "/home/wjchen/Documents/Project/index07.txt"],
                    ["/home/wjchen/tick/CFFEX/T/day/2016/05",
                     "/home/wjchen/tick/CFFEX/T/day/2016/06",
                     "/home/wjchen/tick/CFFEX/T/day/2016/07"], "/home/wjchen/Documents/rhyang/data05-07.h5")
