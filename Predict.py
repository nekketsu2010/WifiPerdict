from keras.engine.saving import load_model
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
# genderNames = ['MA', 'FE']
# generationNames = ['CH', 'AD', 'EL']


directory = "20回目\\"

model = load_model(str(directory) + "Model\\model.ep291_val_loss0.28.hdf5", custom_objects={'f1':f1})

load_array = np.load(str(directory) + 'TestData.npz')
fileNames = np.load("testFileName.npy")
X = load_array['x']

Y = model.predict(X, verbose=1)
# (Y_gender, Y_generation) = model.predict(X, verbose=1)
classes = Y.argmax(axis=-1)
# gender_classes = Y_gender.argmax(axis=-1)
# generation_classes = Y_generation.argmax(axis=-1)
with open(str(directory) + "testPredictResult.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(fileNames)):
        fileName = fileNames[i]
        y_class = classes[i]
        # gender_class = gender_classes[i]
        # generation_class = generation_classes[i]

        writer.writerow([fileName, ClassNames[y_class]])
        # writer.writerow([fileName, genderNames[gender_class] + "_" + generationNames[generation_class]])
