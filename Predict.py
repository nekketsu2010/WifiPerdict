from keras.engine.saving import load_model
import os
import numpy as np
import csv
import keras.backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


ClassNames = ['MA_CH', 'FE_AD', 'MA_AD', 'FE_EL', 'FE_CH', 'MA_EL']

directory = "11回目\\"

model = load_model(str(directory) + "Model\\model.ep242_loss0.03_acc0.99.hdf5", custom_objects={'f1':f1})

load_array = np.load(str(directory) + 'TestData.npz')
fileNames = np.load("testFileName.npy")
X = load_array['x']

Y = model.predict(X, verbose=1)
print(Y)
os.system("EnterKeyください")
classes = Y.argmax(axis=-1)
print(classes)
with open(str(directory) + "testPredictResult.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(fileNames)):
        fileName = fileNames[i]
        _class = classes[i]
        writer.writerow([fileName, ClassNames[_class]])