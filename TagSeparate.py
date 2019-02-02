# 性別と年代のラベルを別扱いにするだけのスクリプト

import pandas as pd
import csv

train_tsv_path = "class_train.tsv"
test_tsv_path = "sample_submit.tsv"

pathes = [train_tsv_path, test_tsv_path]

for path in pathes:
    meta_data = pd.read_table(path)
    print(meta_data)

    # get training dataset and target dataset
    X = list(meta_data.loc[:, "fileName"])
    Y = list(meta_data.loc[:, "target"])

    y_gender = list()
    y_generation = list()

    for y in Y:
        y = y.split('_')
        y_gender.append(y[0])
        y_generation.append(y[1])

    with open("New_" + path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["fileName", "gender", "generation"])
        for i in range(len(X)):
            fileName = X[i]
            gender = y_gender[i]
            generation = y_generation[i]
            writer.writerow([fileName, gender, generation])