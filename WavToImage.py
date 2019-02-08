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
    x, fs = librosa.load(file_path, sr=8000)
    return x, fs

def main(num=0, train=True):
    tsvname = 'New_sample_submit.tsv'
    audioFolder = 'DownSamplingTest'
    imageFolder = 'TestImage'
    if train:
        tsvname = 'New_class_train.tsv'
        audioFolder = 'DownSamplingTrain'
        imageFolder = 'Image'

    meta_data = pd.read_table(tsvname)

    x = list(meta_data.loc[:, "fileName"])
    # save test dataset
    for i in range(len(x)):
        if not os.path.exists(audioFolder + "/" + x[i] + ".wav"):
            print("こいつない")
            print(x[i])
            continue
        if os.path.exists(str(num) + "回目/" + imageFolder + "/" + x[i] + ".png"):
            continue
        _x, fs = load_wave_data(audioFolder, x[i] + ".wav")
        print(str(i) + "個処理しました！")
        #ここからメルスペクトル変換処理
        n_fft = 1024
        hop_length = 128
        stft = np.abs(librosa.stft(_x, n_fft=n_fft, hop_length=hop_length)) ** 2
        log_stft = librosa.power_to_db(stft)
        melsp = librosa.feature.melspectrogram(S=log_stft)
        mfccs = librosa.feature.mfcc(S=melsp)
        mfccs = preprocessing.scale(mfccs, axis=1)
        librosa.display.specshow(mfccs, sr=8000)
        plt.savefig(str(num) + "回目/" + imageFolder + "/" + x[i] + ".png", facecolor="azure", bbox_inches='tight', pad_inches=0)
        plt.close()
    print("終わったお")
