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


if __name__ == '__main__':
    start = datetime.datetime.now()
    dataset = []
    specialset = []

    # 處理問號完後 重新加入資料集
    dataset, specialset = createDataSet(
        dataset, "breast-cancer-wisconsin.data")
    # dataset = np.array(dataset)
    # specialset = np.array(specialset)
    # Handle_question_mark(specialset, dataset[:, 5])
    # dataset = np.vstack((dataset, specialset))
    # dataset = dataset.tolist()
    trainset = []
    dataset = np.array(dataset)
    trainset = dataset[:, :9]

    k1origin = trainset[0]
    k2origin = trainset[1]
    k1 = [0]*9
    k2 = [0]*9
    k1number = 0
    k2number = 0
    classification = [0]*len(trainset)
    breaknumber = 0
    while breaknumber < 9:
        breaknumber = 0
        k1number = 0
        k2number = 0
        k1 = [0]*9
        k2 = [0]*9
        for i in range(len(trainset)):
            first = count_Distance(k1origin, trainset[i])
            second = count_Distance(k2origin, trainset[i])
            if first <= second:
                classification[i] = 2
                k1number += 1
                for j in range(9):
                    k1[j] = k1[j]+trainset[i][j]
            else:
                classification[i] = 4
                k2number += 1
                for j in range(9):
                    k2[j] = k2[j]+trainset[i][j]

        k1next = list(map(lambda x: x/k1number, k1))
        k2next = list(map(lambda x: x/k2number, k2))
        for h in range(9):
            if abs(k1next[h] - k1origin[h]) < 0.001 and abs(k2next[h]-k2origin[h]) < 0.001:
                breaknumber += 1
        k1origin = k1next
        k2origin = k2next
    print("k1中心點=", k1origin, '\n', "k2中心點=", k2origin)
    print("k1數量:", k1number, '\n', "k2數量:", k2number)

    wrongnumberk1 = 0
    wrongnumberk2 = 0
    for i in range(len(classification)):
        if classification[i] != dataset[i][9]:
            if classification[i] == 2:
                wrongnumberk1 += 1
            else:
                wrongnumberk2 += 1
    print("第一群分錯數量:", wrongnumberk1, "第二群分錯數量:", wrongnumberk2)
    print("第一群為2,第二群為4")
