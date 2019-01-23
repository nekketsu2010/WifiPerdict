import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection

# load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=16000)
    return x, fs
# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    # melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=128)
    librosa.display.specshow(mfcc, sr=16000)
    plt.show()
    # return melsp
    return mfcc

# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()

freq = 128
time = 16
def save_np_data(filename, x, y, aug=None, rates=None):
    np_data = np.zeros(freq * time * len(x)).reshape(len(x), freq, time)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x, fs = load_wave_data("train", x[i] + ".wav")
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x)
    #     print(_x.shape)
    #     np_data[i] = _x
    #     np_targets[i] = y[i]
    # np.savez(filename, x=np_data, y=np_targets)

meta_data = pd.read_table("class_train.tsv")
print(meta_data['target'].unique())
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
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, stratify=y)
print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(len(x_train), len(y_train), len(x_test), len(y_test)))

# save test dataset
if not os.path.exists("esc_melsp_test.npz"):
    save_np_data("esc_melsp_test.npz", x_test,  y_test)
# save raw training dataset
if not os.path.exists("esc_melsp_train_raw.npz"):
    save_np_data("esc_melsp_train_raw.npz", x_train,  y_train)