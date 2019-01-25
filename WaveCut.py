import os

import librosa
import pandas as pd
import numpy as np

meta_data = pd.read_table("class_train.tsv")
x = list(meta_data.loc[:, "fileName"])
if not os.path.exists("trainAfter"):
    os.mkdir("trainAfter")
for i in range(len(x)):
    fileName = "train/" + x[i] + ".wav"

    wav, sr = librosa.load(fileName)
    wav, index = librosa.effects.trim(wav, top_db=30)
    print(str(i) + "個のデータ処理完了")
    librosa.output.write_wav("trainAfter/" + x[i] + ".wav", wav,sr=sr, norm=True)