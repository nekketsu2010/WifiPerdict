import math
import os
import wave
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pandas as pd
import gc

directoryName = "test/"
meta_data = pd.read_table("sample_submit.tsv")
x = list(meta_data.loc[:, "fileName"])
# x = x[:30]
# x = ["3042b3c81d6d4ea0165f3ac6c5387272", "3a5c89b172235ee5c45bcb9b5a134971"]

if not os.path.exists("これでどうだ"):
    os.mkdir("これでどうだ")

for i in range(len(x)):
    print(i)
    # wavファイルのデータ取得
    sound = AudioSegment.from_file(str(directoryName) + x[i] + ".wav", format="wav")

    # wavデータの分割（無音部分で区切る）
    chunks = split_on_silence(sound, min_silence_len=200, silence_thresh=-40, keep_silence=250)

    # 分割したデータ毎にファイルに出力
    os.mkdir("これでどうだ/" + x[i])
    for j, chunk in enumerate(chunks):
        chunk.export("これでどうだ/" + x[i] + "/" + str(j) + ".wav", format="wav")