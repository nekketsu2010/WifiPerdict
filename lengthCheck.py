import os
import wave

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import csv

length = []
# load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    # x, fs = librosa.load(file_path, sr=16000)
    wf = wave.open(audio_dir + "/" + file_name + ".wav", "r")
    length.append(float(wf.getnframes()) / wf.getframerate())
    r = float(wf.getnframes()) / wf.getframerate()
    return r


meta_data = pd.read_table("class_train.tsv")
labels, uniques = pd.factorize(meta_data['target'])
meta_data['target'] = labels
print(meta_data)

data_size = meta_data.shape
# arrange target label and its name
class_dict = meta_data["target"].unique()
# print(class_dict)

# get training dataset and target dataset
x = list(meta_data.loc[:, "fileName"])
y = list(meta_data.loc[:, "target"])
with open("trainLength.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(y)):
        len = load_wave_data("train", x[i])
        writer.writerow([x[i], len])
        print(i)
print(max(length))
print(min(length))