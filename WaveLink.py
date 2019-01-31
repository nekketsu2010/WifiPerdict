import wave
import os
import pandas as pd


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
    os.mkdir("testAfter")
    meta_data = pd.read_table("sample_submit.tsv")
    x = list(meta_data.loc[:, "fileName"])
    for i in range(len(x)):
        files = os.listdir("testSeparateData/" + x[i])
        num = 0
        for file in files:
            num += 1
        inputs = ["testSeparateData/" + x[i] + "/" + str(n) + '.wav' for n in range(num)]
        output = "testAfter/" + x[i] + ".wav"
        join_waves(inputs, output)