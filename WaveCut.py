import librosa

fileName = "train/" + ""

wav = librosa.load(fileName)
wav = librosa.effects.trim(wav)

librosa.output.write_wav("test.wav", wav, norm=True)