import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import math


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


def count_Distance(v1, v2):
    sum = 0.0
    for i in range(0, len(v1)):
        sum += math.pow(float(v1[i])-float(v2[i]), 2)
    return sum**0.5


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


def En(attribute1_2, attribute1_4, a3=None):
    if a3 == None:
        a3 = 0
    entropy = np.array([0.0]*10)
    for i in range(len(attribute1_2)):
        total = attribute1_2[i]+attribute1_4[i]
        entropy[i] = caculate_En(attribute1_2[i], total) + \
            caculate_En(attribute1_4[i], total) + \
            caculate_En(a3, total)
    return entropy


def caculate_En(S1, total):
    if(S1 == 0):
        return 0
    return -(S1/total*math.log2(S1/total))


def count_attributes2or4(dataset, i):
    attribute = dataset[:, i]
    tem = dataset[:, 9]
    p_attribute112 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p_attribute114 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for k in range(len(dataset)):
        if tem[k] == 2:
            p_attribute112[attribute[k] -
                           1] = p_attribute112[attribute[k]-1]+1
        if tem[k] == 4:
            p_attribute114[attribute[k] -
                           1] = p_attribute114[attribute[k]-1]+1
    return p_attribute112, p_attribute114


def entropysum(a1, a2, entropy):
    # 計算entropy總和
    anwer = 0
    for i in range(10):
        anwer = anwer + (a1[i]+a2[i])/489*entropy[i]
    return anwer


def findnode(trainset):
    attribute1_2 = np.array([0.0]*10)
    attribute1_4 = np.array([0.0]*10)
    # trainset 489筆資料
    for j in range(10):
        attribute1_2, attribute1_4 = count_attributes2or4(
            trainset, j)
        total2 = attribute1_2.sum()
        entropy = En(attribute1_2, attribute1_4)
        Gain[j] = caculate_En(
            total2, len(trainset))+caculate_En(len(trainset)-total2, len(trainset))-entropysum(attribute1_2, attribute1_4, entropy)
    Gain[9] = 0
    node = np.where(Gain == np.max(Gain))
    return node


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
    accacylist = []
    entropy = np.array([0.0]*10)
    Gain = np.array([0.0]*10)
    for h in range(1):
        trainset, testset = chose_seventity(dataset)
        trainset = np.array(trainset)
        testset = np.array(testset)

        first = findnode(trainset)
        for i in range(9):
            if i != first[0]:
                # G(first,a0)
