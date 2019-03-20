from sklearn.linear_model import LinearRegression
import datetime
import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


def createDataSet(dataset, fileName):
    f = open(fileName, 'r')
    dataset = []
    line = f.readline()
    # 用 while 逐行讀取檔案內容，直至檔案結尾
    while line:
        if line.find('?') == -1:
            attribute = line[:len(line)-1].split(",")
            data = [int(attribute[2]), int(attribute[3]), int(attribute[4]), int(attribute[5]),
                    int(attribute[6]), int(attribute[7]), int(attribute[8]), int(attribute[9])]
            dataset.append(data)
        line = f.readline()
    f.close()
    return dataset


def chose_seventity(dataset):
    # 146筆訓練資料 63筆test資料
    testNumber = len(dataset)-int(len(dataset)*0.7)
    trainset = dataset.copy()
    testset = []
    for i in range(testNumber):
        tem = random.randint(0, len(trainset)-1)
        testset.append(trainset[tem])  # 選出來其中一筆
        del trainset[tem]

    return trainset, testset


if __name__ == '__main__':

    start = datetime.datetime.now()
    dataset = []
    accuracy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dataset = createDataSet(dataset, "machine.data")
    trainset, testset = chose_seventity(dataset)
    trainset = np.array(trainset)
    testset = np.array(testset)

    answer = trainset[:, 7]
    TRAN = trainset[:, :7]
    mean = []
    k = 3
    for i in range(13):
        predict = np.array([])
        neigh = KNeighborsRegressor(n_neighbors=k)
        neigh.fit(TRAN, answer)
        for j in range(len(testset)):
            predict = np.append(predict, neigh.predict([testset[j, :7]]))
        check = testset[:, 7]
        error = (check-predict)/predict
        mean.append(np.sum(error)/len(error))
        k = k+1

    plt.xlabel('K')
    plt.ylabel("error ratio")
    plt.plot(mean)
    end = datetime.datetime.now()
    print(end - start)
    plt.show()
    print('end')
