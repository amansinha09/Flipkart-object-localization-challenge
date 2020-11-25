import pandas as pd
from glob import glob
import pickle as pkl

folder = "./gdrive/My Drive/flipkart/"
train = pd.read_csv(folder+"training.csv")
test = pd.read_csv(folder+"test.csv")
dic = {}

lst = glob('./image/images/*.png')


for i in range((len(test))):
    for j in range(len(lst)):
        print(len(dic))
        if test['image_name'][i] == lst[j].split('/')[1]:
            dic[test['image_name'][i]] = (test['x1'][i], test['x2'][i], test['y1'][i], test['y2'][i])

with open('test.pkl', 'wb') as fh:
    pkl.dump(dic, fh)