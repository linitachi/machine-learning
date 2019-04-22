from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import datetime


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


def normalize(dataset):
    # 正規化資料
    normalize = StandardScaler()
    return normalize.fit_transform(dataset)


def find_Covariancematrix_eigen(dataset):
    # 找出共變異係數矩陣的特徵值
    covmatrix = np.cov(dataset.T)
    eigenvalue, eigenvector = np.linalg.eig(covmatrix)
    return eigenvalue


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

    dataset = normalize(dataset)
    eigenvalue = find_Covariancematrix_eigen(dataset[:, :9])

    pov = []
    totalsum = sum(eigenvalue)
    for i in sorted(eigenvalue, reverse=True):
        pov = np.append(pov, i/totalsum)
    povcum = np.cumsum(pov)

    X = np.linspace(1, 9, 9)
    Y = np.linspace(0, 1, 10)
    plt.bar(X, pov, alpha=0.5, align='center', label='POV')
    plt.step(X, povcum, where='mid', label='POV Cumulative')
    plt.xticks(X)
    plt.yticks(Y)
    plt.xlabel('POV')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
