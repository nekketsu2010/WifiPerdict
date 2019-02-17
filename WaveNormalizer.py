# 音源の音量を正規化するよ
import os
import pandas as pd
import librosa

meta_data = pd.read_table("sample_submit.tsv")
x = list(meta_data.loc[:, "fileName"])

if not os.path.exists("NormalizeTest"):
    os.mkdir("NormalizeTest")

for i in range(len(x)):
    wav, sr = librosa.load("DownSamplingTest/" + x[i] + ".wav", sr=8000)
    wav = librosa.util.normalize(wav)
    librosa.output.write_wav("NormalizeTest/" + x[i] + ".wav", wav, sr=8000)
    print(str(i) + "個終わりました")