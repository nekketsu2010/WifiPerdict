import librosa
import pandas as pd

meta_data = pd.read_table("sample_submit.tsv")
x = list(meta_data.loc[:, "fileName"])

for i in range(len(x)):
    y, sr = librosa.load("silentTest/" + x[i] + ".wav", sr=16000)
    y_8k = librosa.resample(y, sr, 8000)
    librosa.output.write_wav("DownSamplingTest/" + x[i] + ".wav", y_8k, sr=8000, norm=False)
    print(str(i) + "件終わった")
print("end DownSampling")