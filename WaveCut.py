import math
import os
import wave
import librosa
from pydub import AudioSegment
import pandas as pd
import gc

directoryName = "DownSamplingTrain/"
meta_data = pd.read_table("class_train_30.tsv")
x = list(meta_data.loc[:, "fileName"])

for i in range(len(x)):
    print(i)
    if os.path.exists("trainSeparateData/" + x[i]):
        continue
    wav = AudioSegment.from_file(str(directoryName) + x[i] + ".wav", format="wav")
    # wav = AudioSegment.from_file("cash.wav")
    sounds = wav[::30]

    #取り出し終わったあとの処理
    #ゼロクロスしてTrueかFalseの判定
    zeroCrosses = []
    print(sounds)
    for j, sound in enumerate(sounds):#range(len(sounds)):
        # sound = sounds[j]
        sound.export("cash.wav", format="wav")
        sound, _ = librosa.load("cash.wav")
        zeroCross = librosa.zero_crossings(sound, pad=False)
        gc.collect()
        zeroCrosses.append(sum(zeroCross) > 15)
    print(zeroCrosses)
    skip = False
    for j in range(len(zeroCrosses)):
        if skip:
            skip = False
            continue
        if j!=0 and zeroCrosses[j]:
            zeroCrosses[j-1] = True
            if j != len(zeroCrosses)-1:
                zeroCrosses[j+1] = True
                skip = True
    print(zeroCrosses)

    #いよいよWave保存
    num = 0
    sounds = wav[::30]
    print(sounds)
    for j, sound in enumerate(sounds):
        if zeroCrosses[j]:
            print(j)
            print(x[i])
            dirName = "trainSeparateData/" + x[i]
            if not os.path.exists(dirName):
                os.mkdir(dirName)
            sound.export(dirName + "/" + str(num) + ".wav", format="wav")
            gc.collect()
            num += 1

