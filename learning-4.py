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
    v3 = np.corrcoef(v1)
    check = 0
    for i in range(9):
        print('attribute %d和' % (i+2), end=' ')
        for j in range(9):
            if v3[i, j] < 0.5:
                v3[i, j] = 0
            elif i == j:
                v3[i, j] = 0
            else:
                check = 1
                print('%d' %
                      (j+2), end=' ')
        if check == 1:
            print('have strong correlation')
            check = 0
        else:
            print('void have no strong correlation')
    print(np.around(np.corrcoef(v1), 3))
    end = datetime.datetime.now()
    print(end - start)
