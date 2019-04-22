import random
import matplotlib.pyplot as plt
import math
import datetime


def Knn(input_tf, trainset_tf, k):
    tf_distance = dict()
    # 計算每個訓練集合特徵關鍵字字詞頻率向量和輸入向量的距離

    for place in trainset_tf.keys():
        tf_distance[place] = count_Distance(
            trainset_tf.get(place)[:-1], input_tf)

    # 把距離排序，取出k個最近距離的分類

    class_count = dict()
    for i, place in enumerate(sorted(tf_distance, key=tf_distance.get)):
        current_class = trainset_tf.get(place)[-1]
        class_count[current_class] = class_count.get(current_class, 0) + 1

        if (i + 1) >= k:
            break
    # 選出最多的分類 當成最後的分類
    for i, c in enumerate(sorted(class_count, key=class_count.get, reverse=True)):
        if i == 0:
            input_class = c
    return int(input_class)


def createDataSet(dataset, fileName):
    f = open(fileName, 'r')
    dataset = []
    line = f.readline()
    # 用 while 逐行讀取檔案內容，直至檔案結尾

    while line:
        if line.find('?') == -1:
            dataset.append(line[line.find(',')+1:-1])
        line = f.readline()
    return dataset

# 計算兩個點的距離


def count_Distance(v1, v2):
    sum = 0.0
    for i in range(0, len(v1)):
        sum += math.pow(float(v1[i])-float(v2[i]), 2)
    return sum**0.5

# 製造trainset


def createTrainSet(dataset):
    trainset = dict()
    for i in range(len(dataset)):
        tem = dataset[i].split(',')
        Key = i
        trainset[Key] = tem
    return trainset

# 分割dataset 分成70%trainsey 及30%testset


def chose_seventity(dataset):
    # 478筆訓練資料 205筆test資料
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
    dataset = createDataSet(dataset, "breast-cancer-wisconsin.data")
    # j 跑10次
    for j in range(10):
        k = 3
        trainset, testset = chose_seventity(dataset)
        trainset_tf = createTrainSet(trainset)
        # 共205筆資料 做knn
        for i in range(len(testset)):
            s = datetime.datetime.now()
            input_tf = testset[i].split(',')[:-1]
            input_tf = list(map(int, input_tf))
            k = 3
            # k從3~15 跑13次
            for w in range(13):
                answer = Knn(input_tf, trainset_tf, k)
                if answer == int(testset[i][-1]):
                    accuracy[w] = accuracy[w]+1
                k = k+1
    for j in range(13):
        accuracy[j] = accuracy[j]/2050*100
        print("k="+str(j+3)+"的平均正確率: "+str(accuracy[j])+'%')
    plt.xlabel('K')
    plt.ylabel("accuracy(%)")
    plt.plot(range(3, 16), accuracy)
    end = datetime.datetime.now()
    print(end - start)
    plt.show()
