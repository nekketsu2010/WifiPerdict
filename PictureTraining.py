from keras import Sequential, callbacks, layers, Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam, Adamax
import keras.backend as K
import numpy as np
from keras.utils import np_utils
from keras.initializers import he_normal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os

#ClassNames = ['MA_CH', 'FE_AD', 'MA_AD', 'FE_EL', 'FE_CH', 'MA_EL']
genderNames = ['MA', 'FE']
generationNames = ['CH', 'AD', 'EL']
gender_classes = 2
generation_classes = 3

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


def BuildCNN(ipshape=(512, 512, 3)):
    model = Sequential()

    layer0 = layers.Input(ipshape)
    layer1 = layers.Conv2D(24, 3, padding='same', input_shape=ipshape)(layer0)
    layer2 = layers.Activation('relu')(layer1)

    layer3 =layers.Conv2D(48, 3)(layer2)
    layer4 = layers.Activation('relu')(layer3)
    #ここに追記入れ替え
    layer5 = layers.Dropout(0.5)(layer4)
    layer6 = layers.MaxPooling2D(pool_size=(2, 2))(layer5)
    # model.add(Dropout(0.5))

    layer7 = layers.Conv2D(96, 3, padding='same')(layer6)
    layer8 = layers.Activation('relu')(layer7)

    layer9 = layers.Conv2D(96, 3)(layer8)
    layer10 = layers.Activation('relu')(layer9)
    #ここに追記入れ替え、しかも0.8にした
    layer11 = layers.Dropout(0.8)(layer10)
    layer12 = layers.MaxPooling2D(pool_size=(2, 2))(layer11)
    # model.add(Dropout(0.5))

    layer13 = layers.Flatten()(layer12)
    layer14 = layers.Dense(128, kernel_initializer=he_normal())(layer13)
    layer15 = layers.Activation('relu')(layer14)
    # model.add(Dropout(0.5))
    layer16 = layers.BatchNormalization()(layer15)

    #出力層（以下二行で性別・年代に分けて同じモデルで出力できる，softmaxを2つつけた感じ）
    gender_layer = layers.Dense(gender_classes, activation='softmax', name='gender', kernel_initializer=he_normal())(layer16)
    generation_layer = layers.Dense(generation_classes, activation='softmax', name='generation', kernel_initializer=he_normal())(layer16)

    model = Model(inputs=layer0, outputs=[gender_layer, generation_layer])

    adam = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss={"gender": "categorical_crossentropy", "generation": "categorical_crossentropy"},
                  optimizer=adam,
                  metrics=['accuracy', f1])
    model.summary()
    return model

def Learning(num, tsnum=30, nb_epoch=100, batch_size=128, learn_schedule=0.9):
    load_array = np.load(str(num) + '回目/TrainData.npz')
    X = load_array['x']
    Y_gender = load_array['y_gender']
    Y_generation = load_array['y_generation']
    # X = preprocessing.scale(X)

    #one-hot-encoding
    Y_gender = np_utils.to_categorical(Y_gender)
    Y_generation = np_utils.to_categorical(Y_generation)

    # #訓練データとテストデータに分割
    # x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25)


    #訓練
    model = BuildCNN(ipshape=(X.shape[1], X.shape[2], X.shape[3]))
    print(">>　学習開始")

    #nb_epochエポックで１００回学習させる
    for i in range(30):
        #コールバックの設定
        cp_cb = callbacks.ModelCheckpoint(filepath=str(num) + "回目/Model/" + str(i) + "/model.ep{epoch:02d}_loss{loss:.2f}.hdf5", monitor='val_loss', save_best_only=True)
        if not os.path.exists(str(num) + "回目/Model/" + str(i)):
            os.mkdir(str(num) + "回目/Model/" + str(i))
        history = model.fit(X, {'gender': Y_gender, 'generation': Y_generation},
                            batch_size=batch_size,
                            verbose=1,
                            epochs=nb_epoch,
                            shuffle=True,
                            validation_split=0.25,
                            callbacks=[cp_cb])
        p = np.random.permutation(len(X))
        X = X[p]
        Y_gender = Y_gender[p]
        Y_generation = Y_generation[p]

def main(num=0):
    Learning(num)