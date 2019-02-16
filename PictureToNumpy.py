import glob
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

hw = {"height":50, "width":38}
def PreProcess(dirname, train):
    # Gender x Generation
    # tsvname = 'New_sample_submit.tsv'
    tsvname = 'sample_submit.tsv'
    imageFolder = 'TestImage'
    if train:
        # Gender x Generation
        # tsvname = 'New_class_train.tsv'
        tsvname = 'class_train.tsv'
        imageFolder = 'Image'

    arrlist = []
    meta_data = pd.read_table(tsvname)
    # gender_labels, uniques = pd.factorize(meta_data['gender'])
    # generation_labels, uniques = pd.factorize(meta_data['generation'])
    labels, uniques = pd.factorize(meta_data['target'])
    meta_data['target'] = labels
    # meta_data['gender'] = gender_labels
    # meta_data['generation'] = generation_labels
    x = list(meta_data.loc[:, "fileName"])
    y = list(meta_data.loc[:, 'target'])
    # y_gender = list(meta_data.loc[:, 'gender'])
    # y_generation = list(meta_data.loc[:, 'generation'])
    # np_genders = np.zeros(len(y_gender))
    # np_generations = np.zeros(len(y_generation))
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        if not os.path.exists(str(dirname) + "\\" + imageFolder + "\\" + x[i] + ".png"):
            continue
        #史上初！グレースケール　←やめた
        img = load_img(str(dirname) + "\\" + imageFolder + "\\" + x[i] + ".png", target_size=(hw["height"], hw["width"]))  # 画像ファイルの読み込み
        array = img_to_array(img) / 255  # 画像ファイルのnumpy化
        #追記
        array_list = array.tolist()
        arrlist.append(array_list)  # numpy型データをリストに追加
        np_targets[i] = y[i]
        # np_genders[i] = y_gender[i]
        # np_generations[i] = y_generation[i]
        print("%d個のデータを処理しました" % i)
    nplist = np.asarray(arrlist)
    print(">> " + dirname + "から" + str(i) + "個ファイル読み込み成功")
    if train:
        np.savez(str(dirname) + "\\TrainData", x=nplist, y=np_targets)
        # np.savez(str(dirname) + "\\TrainData", x=nplist, y_gender=np_genders, y_generation=np_generations)
    else:
        np.savez(str(dirname) + "\\TestData", x=nplist, y=np_targets)
        # np.savez(str(dirname) + "\\TestData", x=nplist, y_gender=np_genders, y_generation=np_generations)

def main(num=0, train=True):
    PreProcess(str(num) + "回目", train)