from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
def build_multilayer_perceptron():
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
    model.add(Dense(10, input_shape=(2740, )))  # 入力層4ノード, 隠れ層に10ノード, 全結合
    model.add(Activation("sigmoid"))  # 活性化関数はsigmoid
    model.add(Dense(14))  # 出力層3ノード,全結合
    model.add(Activation("sigmoid"))
    return model

def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)

    zscore = (x-xmean)/xstd
    return zscore

def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / ( x_max - x_min)
    return x_norm

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="loss for training")
    axR.plot(fit.history['val_acc'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='lower right')

# Irisデータをロード
dataset = pd.read_csv("wifidata5GHz.csv")
print(dataset)
dataset = dataset.sample(frac=1)
X = DataFrame(dataset.drop("RoomName", axis=1))
Y = DataFrame(dataset["RoomName"])
# X = min_max_normalization(X)
# X = zscore(X)
X = X.astype(np.double)
X = preprocessing.scale(X)
Y = pd.get_dummies(Y, columns=["RoomName"])
Y = Y.astype(np)
print(Y.shape)
print(Y.nunique())
#訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# データの標準化
x_train = x_train.astype(np.double)
x_test = x_test.astype(np.double)
x_train, x_valid = np.split(x_train, [2500])
y_train, y_valid = np.split(y_train, [2500])
# y_train = y_train.astype(np.unicode)
print(y_train)

# ラベルをone-hot-encoding形式に変換
# 0 => [1, 0, 0]
# 1 => [0, 1, 0]
# 2 => [0, 0, 1]
# Y = np_utils.to_categorical(Y)


# モデル構築
model = build_multilayer_perceptron()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデル訓練
fit = model.fit(x_train, y_train, epochs=200, batch_size=128, shuffle=True, verbose=1, validation_data=(x_valid, y_valid))
plot_history_loss(fit)
plot_history_acc(fit)
plt.show()
# モデル評価
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
predict_classes = model.predict_classes(x_test, batch_size=32)
print(predict_classes)
true_classes = np.argmax(y_test.values,axis=1)
print(true_classes)
print("Loss = {:.2f}".format(loss))
print("Accuracy = {:.2f}".format(accuracy))

print(confusion_matrix(true_classes, predict_classes))
