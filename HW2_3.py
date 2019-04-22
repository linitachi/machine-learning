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

# 算出所有可能的機率


def count_probability(dataset):
    p2 = 0
    p4 = 0
    attribute11 = dataset[:, 9]
    for i in range(len(dataset)):
        if attribute11[i] == 2:
            p2 = p2+1
        else:
            p4 = p4+1

    p_attribute = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pa2ofconditional2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pa2ofconditional4 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(9):
        p_attribute[i] = count_attributes(dataset, i)
        pa2ofconditional2[i], pa2ofconditional4[i] = count_attributes2or4(
            dataset, i, p2, p4)

    p2 = p2/len(dataset)
    p4 = p4/len(dataset)
    return p2, p4, p_attribute, pa2ofconditional2, pa2ofconditional4


def count_attributes(dataset, i):
    attribute = dataset[:, i]
    p_attribute = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for k in range(len(dataset)):
        p_attribute[attribute[k] -
                    1] = p_attribute[attribute[k]-1]+1/len(dataset)
    return p_attribute


def count_attributes2or4(dataset, i, p2, p4):
    attribute = dataset[:, i]
    tem = dataset[:, 9]
    p_attribute112 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_attribute114 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for k in range(len(dataset)):
        if tem[k] == 2:
            p_attribute112[attribute[k] -
                           1] = p_attribute112[attribute[k]-1]+1/p2
        if tem[k] == 4:
            p_attribute114[attribute[k] -
                           1] = p_attribute114[attribute[k]-1]+1/p4
    return p_attribute112, p_attribute114


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
    for h in range(10):
        trainset, testset = chose_seventity(dataset)
        trainset = np.array(trainset)
        testset = np.array(testset)
        p2, p4, p_attribute, pa2ofconditional2, pa2ofconditional4 = count_probability(
            trainset)
        accacy = 0

        for i in testset:
            final2 = 1
            final4 = 1
            for j in range(9):
                if p_attribute[j][i[j]-1] != 0:
                    final2 *= p2 * \
                        pa2ofconditional2[j][i[j]-1]/p_attribute[j][i[j]-1]
                    final4 *= p2 * \
                        pa2ofconditional4[j][i[j]-1]/p_attribute[j][i[j]-1]
            if final2 >= final4:
                decide = 2
            else:
                decide = 4
            if decide == i[9]:
                accacy = accacy+1
        accacylist.append(accacy/len(testset))
        print(accacy/len(testset))
    end = datetime.datetime.now()
    print(end - start)

    plt.xlabel('times')
    plt.ylabel("accurcy")
    plt.plot(range(1, 11), accacylist)
    plt.show()
