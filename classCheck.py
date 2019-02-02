#　性別・年代のラベルを番号に直したときの対応がわからないので調べるやつ

import pandas as pd
import numpy as np

meta_data = pd.read_table("New_class_train.tsv")
labels, uniques = pd.factorize(meta_data['generation'])
meta_data['generation'] = labels

print(uniques)