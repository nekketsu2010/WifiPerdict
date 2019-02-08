# RNNに使う行列を音データから作成する

import pandas as pd

tsvname = "class_train.tsv"

meta_data = pd.read_table(tsvname)
x = list(meta_data.loc[:, "fileName"])

data_num = len(x)
