# 行列を縦につなげるよ

import numpy as np

num = 23
splitN = 6
folderName = str(num) + "回目"
numpyNames = ['TrainData_wn.npz', 'TrainData_ss.npz', 'TrainData_st.npz', 'TrainData_com.npz']

j = 0
for i in range(6):
    matrix = np.load(str(folderName) + '/TrainData.npz')
    x = matrix['x']
    y = matrix['y']
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
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
        p = np.random.permutation(len(X))
        x = x[p]
        y = y[p]

        for xn in range(len(X)):
            X[xn] = np.vstack((X[xn], x[xn]))
            Y[xn] = np.hstack((Y[xn], y[xn]))
            print(X[xn].shape)
            print(Y[xn].shape)
    for xn in range(len(X)):
        np.savez(str(folderName) + "\\TrainData" + str(j), x=X[xn], y=Y[xn])
        j += 1
