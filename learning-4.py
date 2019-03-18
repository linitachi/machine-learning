import random
import matplotlib.pyplot as plt
import math
import datetime
import numpy as np


def createDataSet(dataset, fileName):
    f = open(fileName, 'r')
    dataset = []
    line = f.readline()
    # 用 while 逐行讀取檔案內容，直至檔案結尾

    while line:
        if line.find('?') == -1:
            attribute = line[:len(line)-1].split(",")
            data = (int(attribute[1]), int(attribute[2]), int(attribute[3]), int(attribute[4]), int(attribute[5]),
                    int(attribute[6]), int(attribute[7]), int(attribute[8]), int(attribute[9]))
            dataset.append(data)
        line = f.readline()
    f.close()
    return dataset


if __name__ == '__main__':

    start = datetime.datetime.now()
    dataset = []
    accuracy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dataset = createDataSet(dataset, "breast-cancer-wisconsin.data")
    v1 = np.array(dataset).T

    v2 = np.cov(v1)
    v3 = np.corrcoef(v2)
    for i in range(9):
        for j in range(9):
            if v3[i, j] < 0.5:
                v3[i, j] = 0
            elif i == j:
                v3[i, j] = 0
    print(np.around(v3, 3))
    end = datetime.datetime.now()
    print(end - start)
