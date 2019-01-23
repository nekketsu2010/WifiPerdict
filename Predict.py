from keras.engine.saving import load_model
import numpy as np
import csv

ClassNames = ['MA_CH', 'FE_AD', 'MA_AD', 'FE_EL', 'FE_CH', 'MA_EL']

directory = "4回目\\"

model = load_model(str(directory) + "Model\\model.ep2992_loss0.04_acc0.99.hdf5")

load_array = np.load(str(directory) + 'TestData.npz')
fileNames = np.load("testFileName.npy")
X = load_array['x']

Y = model.predict(X, verbose=1)
classes =  Y.argmax(axis=-1)
with open(str(directory) + "testPredictResult.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(fileNames)):
        fileName = fileNames[i]
        _class = classes[i]
        writer.writerow([fileName, ClassNames[_class]])