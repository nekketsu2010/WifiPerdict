from keras import Sequential, callbacks
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam, Adamax
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

ClassNames = ['MA_CH', 'FE_AD', 'MA_AD', 'FE_EL', 'FE_CH', 'MA_EL']
def BuildCNN(ipshape=(32, 32, 3), num_classes=6):
    model = Sequential()

    model.add(Conv2D(24, 3, padding='same', input_shape=ipshape))
    model.add(Activation('relu'))

    model.add(Conv2D(48, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(96, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(96, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    adam = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

def Learning(num, tsnum=30, nb_epoch=3000, batch_size=256, learn_schedule=0.9):
    load_array = np.load(str(num) + '回目/TrainData.npz')
    X = load_array['x']
    Y = load_array['y']
    # X = preprocessing.scale(X)

    #one-hot-encoding
    Y = np_utils.to_categorical(Y)

    # #訓練データとテストデータに分割
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    #コールバックの設定
    cp_cb = callbacks.ModelCheckpoint(filepath=str(num) + "回目/Model/model.ep{epoch:02d}_loss{loss:.2f}_acc{acc:.2f}.hdf5", monitor='loss', save_best_only=True)

    #訓練
    model = BuildCNN(ipshape=(X.shape[1], X.shape[2], X.shape[3]), num_classes=6)
    print(">>　学習開始")
    history = model.fit(X, Y,
                        batch_size=batch_size,
                        verbose=1,
                        epochs=nb_epoch,
                        callbacks=[cp_cb])

def main(num=0):
    Learning(num)