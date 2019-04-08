import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from itertools import combinations


def createDataSet(dataset, fileName):
    f = open(fileName, 'r')
    dataset = []
    specialset = []
    line = f.readline()
    # 用 while 逐行讀取檔案內容，直至檔案結尾
    while line:
        if line.find('?') == -1:
            attribute = line[:len(line)-1].split(",")
            # 10 attribute
            data = [int(attribute[1]), int(attribute[2]), int(attribute[3]), int(attribute[4]), int(attribute[5]),
                    int(attribute[6]), int(attribute[7]), int(attribute[8]), int(attribute[9]), int(attribute[10])]
            dataset.append(data)
        else:
            attribute = line[:len(line)-1].split(",")
            # 10 attribute
            for i in range(10):
                if attribute[i+1] == "?":
                    attribute[i+1] = -1
            data = [int(attribute[1]), int(attribute[2]), int(attribute[3]), int(attribute[4]), int(attribute[5]),
                    int(attribute[6]), int(attribute[7]), int(attribute[8]), int(attribute[9]), int(attribute[10])]
            specialset.append(data)
        line = f.readline()
    f.close()
    return dataset, specialset


def Handle_question_mark(s, attribute):
    s[:, 5] = attribute.mean()
    return s


def chose_seventity(dataset):
    # 489筆訓練資料 210筆test資料
    testNumber = len(dataset)-int(len(dataset)*0.7)
    trainset = dataset.copy()
    testset = []
    for i in range(testNumber):
        tem = random.randint(0, len(trainset)-1)
        testset.append(trainset[tem])  # 選出來其中一筆
        del trainset[tem]

    return trainset, testset


def chose_fivety(dataset):
    # 489筆訓練資料
    testNumber = len(dataset)-int(len(dataset)*5/7)
    trainset = dataset.copy()
    testset = []
    for i in range(testNumber):
        tem = random.randint(0, len(trainset)-1)
        testset.append(trainset[tem])  # 選出來其中一筆
        del trainset[tem]

    return trainset, testset


def findbestattribute(trainset, findattributeset):
    # 共84種可能
    chose = list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], 3))
    accacylist = [0]*84

    for i in range(len(findattributeset)):
        w = datetime.datetime.now()
        ii = 0
        for l, j, h in chose:
            accacylist[ii] = accacylist[ii] + \
                Knn(trainset, findattributeset[i], 3, l, j, h)
            ii = ii+1
        s = datetime.datetime.now()
        print(s-w)
    for i in range(len(accacylist)-1):
        if accacylist[i] > accacylist[i+1]:
            maxs = i
        else:
            maxs = i+1
    end = datetime.datetime.now()
    print(end-start)
    return chose[maxs]
    # for i, j, k in chose:
    #     for trainsetn in range(len(trainset)):
    #         print(i, j, k)


def Knn(trainset, test, k, l, j, h):
    distance_tem = np.array([])
    # print(trainset[:, l][0])
    # print(test[9])
    for i in range(len(trainset)):
        disnp = np.array(
            [trainset[:, l][i], trainset[:, j][i], trainset[:, h][i]])
        testnp = np.array([test[l], test[j], test[h]])
        distance_tem = np.append(
            distance_tem, [count_Distance(disnp, testnp)], axis=0)
    print(np.argsort(distance_tem)[2])
    minset = []

    for i in range(3):
        minv = np.where(distance_tem == np.min(distance_tem))
        if(len(minv[0]) >= 3):
            minset = np.array([minv[0][0], minv[0][1], minv[0][2]])
            break
        elif(len(minv[0]) == 2):
            distance_tem[minv[0][0]] = 999999999
            distance_tem[minv[0][1]] = 999999999
            minset = np.append(
                minset, [minv[0][0], minv[0][1]], axis=0)
        else:
            distance_tem[minv[0][0]] = 999999999
            minset = np.append(
                minset, [minv[0][0]], axis=0)
    minset = list(map(int, minset))
    # 找出答案
    if trainset[:, 9][minset[0]] == trainset[:, 9][minset[1]]:
        anwer = trainset[:, 9][minset[0]]
    elif trainset[:, 9][minset[0]] == trainset[:, 9][minset[2]]:
        anwer = trainset[:, 9][minset[0]]
    else:
        anwer = trainset[:, 9][minset[1]]
    if anwer == test[9]:
        return 1
    else:
        return 0


def count_Distance(v1, v2):
    sum = 0.0
    for i in range(0, len(v1)):
        sum += math.pow(float(v1[i])-float(v2[i]), 2)
    return sum**0.5


if __name__ == '__main__':

    start = datetime.datetime.now()
    dataset = []
    specialset = []

    # 處理問號完後 重新加入資料集
    dataset, specialset = createDataSet(
        dataset, "breast-cancer-wisconsin.data")
    dataset = np.array(dataset)
    specialset = np.array(specialset)
    Handle_question_mark(specialset, dataset[:, 5])
    dataset = np.vstack((dataset, specialset))
    dataset = dataset.tolist()

    accacylist = [0]*10
    for h in range(10):
        trainset, testset = chose_seventity(dataset)
        trainset, findattributeset = chose_fivety(trainset)
        trainset = np.array(trainset)
        testset = np.array(testset)
        findattributeset = np.array(findattributeset)
        correctlist = findbestattribute(trainset, findattributeset)

        for i in range(len(testset)):
            accacylist[h] = accacylist[h]+Knn(trainset, testset[i], 3, correctlist[0],
                                              correctlist[1], correctlist[2])/len(testset)
        print("第"+str(h))
    end = datetime.datetime.now()
    print(end - start)
    plt.xlabel('times')
    plt.ylabel("accurcy")
    plt.plot(range(1, 11), accacylist)
    plt.show()
