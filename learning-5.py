import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import math


def createDataSet(dataset, fileName):
    f = open(fileName, 'r')
    dataset = []
    line = f.readline()
    # 用 while 逐行讀取檔案內容，直至檔案結尾
    while line:
        if line.find('?') == -1:
            attribute = line[:len(line)-1].split(",")
            # 8 attribute
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


def Knn_regression_predict(TRAN, answer, test, k):
    tem = answer.copy()
    distance_tem = np.array([])
    distance = 0
    for i in range(len(TRAN)):
        distance_tem = np.append(
            distance_tem, [count_Distance(TRAN[i], test)], axis=0)

    for i in range(k):
        minv = np.where(distance_tem == np.min(distance_tem))
        distance = distance+tem[minv[0][0]]
        distance_tem = np.delete(distance_tem, minv[0][0])
        tem = np.delete(tem, minv[0][0])

    return distance/k


def count_Distance(v1, v2):
    sum = 0.0
    for i in range(0, len(v1)):
        sum += math.pow(float(v1[i])-float(v2[i]), 2)
    return sum**0.5


def MSE_function(answer, test):
    Sum = 0.0
    Sum += (answer-test)**2
    Sum = np.sum(Sum)
    Sum = Sum/len(answer)
    return Sum


if __name__ == '__main__':

    start = datetime.datetime.now()
    dataset = []
    dataset = createDataSet(dataset, "machine.data")
    dataset = np.array(dataset)
    # calculate mean
    X_mean = dataset.mean(axis=0)
    # calculate variance
    X_std = dataset.std(axis=0)
    # standardize X
    X_scaled = (dataset-X_mean)/X_std
    X_scaled = X_scaled.tolist()
    MSE = [0] * 13
    r2Score = [0] * 13
    times = 10

    for time in range(times):
        k = 3
        trainset, testset = chose_seventity(X_scaled)
        trainset = np.array(trainset)
        testset = np.array(testset)
        answer = trainset[:, 6]  # attribute 9
        TRAN = np.delete(trainset, np.s_[6: 8], axis=1)  # 砍掉 attribute 9 10
        for i in range(13):
            predict = np.array([])
            test = np.delete(testset, np.s_[6: 8], axis=1)
            for j in range(len(testset)):
                predict = np.append(
                    predict, Knn_regression_predict(TRAN, answer, test[j, ], k))
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

            error = MSE_function(check, predict)
            # r2_score
            score = 1-(np.sum((check-predict)**2)) / \
                np.sum((check - np.mean(check))**2)

            r2Score[i] = r2Score[i]+score
            MSE[i] = MSE[i]+error
            k = k+1
    MSE = [c/times for c in MSE]
    r2Score = [c/times for c in r2Score]

    plt.xlabel('K')
    plt.ylabel("Mean squared error ")
    plt.plot(range(3, 16), MSE)

    end = datetime.datetime.now()
    print(end - start)
    plt.show()

    plt.xlabel('K')
    plt.ylabel("r2_score")
    plt.plot(range(3, 16), r2Score)
    plt.show()
    print('end')
