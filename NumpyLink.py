# 行列を縦につなげるよ

import numpy as np

num = 23
splitN = 6
folderName = str(num) + "回目"
numpyNames = ['TrainData_wn.npz', 'TrainData_ss.npz', 'TrainData_st.npz', 'TrainData_com.npz']

matrix = np.load(str(folderName) + '/TrainData.npz')
x = matrix['x']
y = matrix['y']
print(x.shape)
print(y.shape)
X = np.split(x, splitN)
Y = np.split(y, splitN)
print(X[0].shape)
print(Y[0].shape)

for numpyNum in range(len(numpyNames)):
    matrix = np.load(str(folderName) + "/" + numpyNames[numpyNum])
    x = matrix['x']
    y = matrix['y']
    x = np.split(x, splitN)
    y = np.split(y, splitN)    

    for xn in range(len(X)):
        X[xn] = np.vstack((X[xn], x[xn]))
        Y[xn] = np.hstack((Y[xn], y[xn]))
        print(X[xn].shape)
        print(Y[xn].shape)
for xn in range(len(X)):
    np.savez(str(folderName) + "\\TrainData" + str(xn), x=X[xn], y=Y[xn])
