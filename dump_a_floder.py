import pickle
import os
path = ''
files = os.listdir(path)
csvfiles = [f for f in files if f.endswith('csv')]
pickle.dump(csvfiles,'index05.txt')
