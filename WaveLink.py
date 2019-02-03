import wave
import os
import pandas as pd
import shutil


def join_waves(inputs, output):
    '''
    inputs : list of filenames
    output : output filename
    '''
    try:
        fps = [wave.open(f, 'r') for f in inputs]
        fpw = wave.open(output, 'w')

        fpw.setnchannels(fps[0].getnchannels())
        fpw.setsampwidth(fps[0].getsampwidth())
        fpw.setframerate(fps[0].getframerate())

        for fp in fps:
            fpw.writeframes(fp.readframes(fp.getnframes()))
            fp.close()
        fpw.close()

    except Exception as e:
        print(e)


if __name__ == '__main__':
    if not os.path.exists("どうよ"):
        os.mkdir("どうよ")
    src = "test/"
    copy = "どうよ/"
    meta_data = pd.read_table("sample_submit.tsv")
    x = list(meta_data.loc[:, "fileName"])

    # x = ["3042b3c81d6d4ea0165f3ac6c5387272", "3a5c89b172235ee5c45bcb9b5a134971"]
    for i in range(len(x)):
        folder = "これでどうだ/"
        if os.path.exists(folder + x[i]):
            files = os.listdir(folder + x[i])
            num = 0
            for file in files:
                num += 1

            if num==0:
                #num=0→オリジナルのファイルをコピーすることにする
                shutil.copy(src + x[i] + ".wav", copy + x[i] + ".wav")
                continue
            inputs = [folder + x[i] + "/" + str(n) + '.wav' for n in range(num)]
            output = "どうよ/" + x[i] + ".wav"
            if not os.path.exists(output):
                join_waves(inputs, output)