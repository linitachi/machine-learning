from sklearn.linear_model import LinearRegression
import datetime
import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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
    dataset = createDataSet(dataset, "machine.data")
    final = [0] * 13
    r2Score = [0] * 13
    times = 10

    for time in range(times):
        k = 3
        trainset, testset = chose_seventity(dataset)
        trainset = np.array(trainset)
        testset = np.array(testset)
        answer = trainset[:, 6]
        TRAN = np.delete(trainset, np.s_[6:8], axis=1)
        mean = []
        for i in range(13):
            predict = np.array([])
            neigh = KNeighborsRegressor(n_neighbors=k)
            neigh.fit(TRAN, answer)
            test = np.delete(testset, np.s_[6:8], axis=1)
            for j in range(len(testset)):
                predict = np.append(predict, neigh.predict([test[j, ]]))
            check = testset[:, 6]
            # 把極端值忽略
            sub = predict-check
            for q in range(5):
                maxv = np.where(sub == np.max(sub))
                sub = np.delete(sub, maxv)
                predict = np.delete(predict, maxv)
                check = np.delete(check, maxv)

                minv = np.where(sub == np.min(sub))
                sub = np.delete(sub, minv)
                predict = np.delete(predict, minv)
                check = np.delete(check, minv)

            error = mean_squared_error(check, predict)
            score = r2_score(check, predict)
            r2Score[i] = r2Score[i]+score
            final[i] = final[i]+error
            k = k+1
    final = [c/times for c in final]
    r2Score = [c/times for c in r2Score]
    plt.xlabel('K')
    plt.ylabel("Mean squared error ")
    plt.plot(range(3, 16), final)
    end = datetime.datetime.now()
    print(end - start)
    plt.show()

    plt.xlabel('K')
    plt.ylabel("r2_score")
    plt.plot(range(3, 16), r2Score)
    plt.show()
    print('end')
