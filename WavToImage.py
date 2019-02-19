import os
import random

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

# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


def main(num=0, train=True):
    tsvname = 'sample_submit.tsv'
    # audioFolder = 'DownSamplingTest'
    audioFolder = 'NormalizeTest'
    imageFolders = ['TestImage']
    if train:
        tsvname = 'class_train.tsv'
        # audioFolder = 'DownSamplingTrain'
        audioFolder = 'NormalizeTrain'
        imageFolders = ['Image', 'Image_wn', 'Image_ss', 'Image_st', 'Image_com']

    meta_data = pd.read_table(tsvname)

    x = list(meta_data.loc[:, "fileName"])
    n_fft = 4096
    hop_length = 1024  # n_fftとってきてhop_lenghtだけずらしてn_fftだけポイントを取ってくる
    for image_num in range(len(imageFolders)):
        if image_num == 0 or image_num == 1 or image_num == 2:
            continue
        imageFolder = imageFolders[image_num]
        for i in range(len(x)):
            if not os.path.exists(audioFolder + "/" + x[i] + ".wav"):
                print("こいつない")
                print(x[i])
                continue
            if os.path.exists(str(num) + "回目/" + imageFolder + "/" + x[i] + ".png"):
                continue
            _x, fs = load_wave_data(audioFolder, x[i] + ".wav")

            #ここで，numの値に応じたノイズを加える（num>0なら）
            if num == 1:
                rates = random.randint(1, 50) / 10000
                _x = add_white_noise(_x, rate=rates)
            elif num == 2:
                _x = shift_sound(_x, rate=random.randint(2, 6))
            elif num == 3:
                _x = stretch_sound(_x, rate=random.randint(80, 120)/100)
            elif num == 4:
                rates = random.randint(1, 50) / 10000
                _x = add_white_noise(_x, rate=rates)
                if random.choice([True, False]):
                    _x = shift_sound(_x, rate=random.randint(2, 6))
                else:
                    _x = stretch_sound(_x, rate=random.randint(80, 120) / 100)

            #ここからメルスペクトル変換処理
            stft = np.abs(librosa.stft(_x, n_fft=n_fft, hop_length=hop_length)) ** 2
            log_stft = librosa.power_to_db(stft)
            librosa.display.specshow(log_stft, sr=8000)
            plt.savefig(str(num) + "回目/" + imageFolder + "/" + x[i] + ".png", facecolor="azure", bbox_inches='tight', pad_inches=0)
            plt.close()
            # 二枚目が一番わかり易い↑
            # melsp = librosa.feature.melspectrogram(S=log_stft)
            # mfccs = librosa.feature.mfcc(S=melsp)
            # mfccs = preprocessing.scale(mfccs, axis=1)
            # librosa.display.specshow(melsp, sr=8000, x_axis='time', y_axis='mel')
            # plt.savefig(str(num) + "回目/" + imageFolder + "/" + x[i] + ".png", facecolor="azure", bbox_inches='tight', pad_inches=0)
            # plt.show()
            # plt.close()
            print(str(i) + "個処理しました！ " + imageFolder)
    print("終わったお")