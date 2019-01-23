import glob
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import load_img

hw = {"height":64, "width":48}
def PreProcess(dirname):
    arrlist = []
    meta_data = pd.read_table("sample_submit.tsv")
    labels, uniques = pd.factorize(meta_data['target'])
    meta_data['target'] = labels
    x = list(meta_data.loc[:, "fileName"])
    y = list(meta_data.loc[:, 'target'])
    np.save("testFileName", x)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        img = load_img(str(dirname) + "\\TestImage\\" + x[i] + ".png", target_size=(hw["height"], hw["width"]))  # 画像ファイルの読み込み
        array = img_to_array(img) / 255  # 画像ファイルのnumpy化
        #追記
        array_list = array.tolist()
        arrlist.append(array_list)  # numpy型データをリストに追加
        np_targets[i] = y[i]
        print("%d個のデータを処理しました" % i)
    nplist = np.asarray(arrlist)
    print(">> " + dirname + "から" + str(i) + "個ファイル読み込み成功")
    np.savez(str(dirname) + "\\TestData", x=nplist, y=np_targets)

def main(num=0):
    PreProcess(str(num) + "回目")