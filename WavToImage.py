import os

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# load a wave data
from sklearn import preprocessing


def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=16000)
    return x, fs

# change wave data to mel-stft
def calculate_melsp(x, filename, num, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    # melsp = preprocessing.scale(melsp, axis=1)
    # mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=128)
    # spec = librosa.feature.melspectrogram(x, sr=16000)
    # librosa.display.specshow(mfcc, sr=16000)
    librosa.display.specshow(melsp, sr=16000)
    plt.savefig(str(num) + "回目/Image/" + filename + ".png")
    plt.close()
    # return melsp
    # return mfcc

def save_np_data(num, x, y, aug=None, rates=None):

    for i in range(len(y)):
        _x, fs = load_wave_data("trainAfter", x[i] + ".wav")
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        print(str(i) + "個処理しました！")
        _x = calculate_melsp(_x, x[i], num)
    #     print(_x.shape)
    #     np_data[i] = _x
    #     np_targets[i] = y[i]
    # np.savez(filename, x=np_data, y=np_targets)

def main(num=0):
    meta_data = pd.read_table("class_train.tsv")
    labels, uniques = pd.factorize(meta_data['target'])
    meta_data['target'] = labels
    print(meta_data)

    data_size = meta_data.shape
    # arrange target label and its name
    class_dict = meta_data["target"].unique()
    print(class_dict)

    # get training dataset and target dataset
    x = list(meta_data.loc[:, "fileName"])
    y = list(meta_data.loc[:, "target"])
    # save test dataset
    if not os.path.exists("esc_melsp_test.npz"):
        save_np_data(num, x, y)
    print("終わったお")