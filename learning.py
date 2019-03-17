import random
import math
import matplotlib.pyplot as plt


def Knn(input_tf, trainset_tf, trainset_class, k):
    tf_distance = dict()
    # 計算每個訓練集合特徵關鍵字字詞頻率向量和輸入向量的距離

    # print('(1) 計算向量距離')
    for place in trainset_tf.keys():
        tf_distance[place] = count_Distance(trainset_tf.get(place), input_tf)
        # print('\tTF(%s) = %f' % (place, tf_distance.get(place)))

    # 把距離排序，取出k個最近距離的分類

    class_count = dict()
    # print('(2) 取K個最近鄰居的分類, k = %d' % k)
    for i, place in enumerate(sorted(tf_distance, key=tf_distance.get)):
        current_class = trainset_class.get(place)
        # print('\tTF(%s) = %f, class = %s' %(place, tf_distance.get(place), current_class))
        class_count[current_class] = class_count.get(current_class, 0) + 1
        if (i + 1) >= k:
            break
    # print('(3) K個最近鄰居分類出現頻率最高的分類當作最後分類')
    input_class = ''
    for i, c in enumerate(sorted(class_count, key=class_count.get, reverse=True)):
        if i == 0:
            input_class = c
        # print('\t%s, %d' % (c, class_count.get(c)))
    # print('(4) 分類結果 = %s' % input_class)
    return int(input_class)


def createDataSet(dataset, fileName):
    f = open(fileName, 'r')
    dataset = []
    line = f.readline()
    # 用 while 逐行讀取檔案內容，直至檔案結尾
    while line:
        if line.find('?') == -1:
            dataset.append(line[0:-1])
        line = f.readline()
    return dataset


def count_Distance(v1, v2):
    sum = 0.0
    for i in range(0, len(v1)):
        sum += math.pow(float(v1[i])-float(v2[i]), 2)
    return sum**0.5


def createTrainSet(dataset):
    trainset_tf = dict()
    trainset_class = dict()
    for i in range(len(dataset)):
        tem = dataset[i].split(',')
        print(tem)
        Key = i
        Value = tem[len(tem)-1]
        trainset_tf[Key] = tem[1:-1]
        trainset_class[Key] = Value
    return trainset_tf, trainset_class


def chose_seventity(dataset):
    trainNumber = int(len(dataset)*0.7)  # 478筆訓練資料 205筆test資料
    testNumber = len(dataset)-trainNumber
    trainset = dataset.copy()
    testset = []
    for i in range(testNumber):
        tem = random.randint(0, len(trainset)-1)
        testset.append(trainset[tem])  # 選出來其中一筆
        del trainset[tem]
    return trainset, testset


def generatetestset(testset):
    newset = []
    tem = ''
    for j in range(len(testset)):
        string = testset[j].split(',')
        for i in range(len(string)-2):
            tem = tem+string[i+1]+','
        tem = tem[0:-1]
        newset.append(tem)
        tem = ''
    return newset


if __name__ == '__main__':

    dataset = []
    accuracy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dataset = createDataSet(dataset, "breast-cancer-wisconsin.data")
    for j in range(10):
        k = 3
        trainset, checkset = chose_seventity(dataset)
        trainset_tf, trainset_class = createTrainSet(trainset)
        testset = generatetestset(checkset)
        for i in range(len(testset)):
            input_tf = testset[i].split(',')
            input_tf = list(map(float, input_tf))
            k = 3
            for w in range(13):
                answer = Knn(input_tf, trainset_tf, trainset_class, k)
                if answer == int(checkset[i][-1]):
                    accuracy[w] = accuracy[w]+1
                k = k+1
    for j in range(13):
        accuracy[j] = accuracy[j]/2050*100
        print("k="+str(j+3)+"的平均正確率: "+str(accuracy[j])+'%')
    plt.xlabel('K')
    plt.ylabel("accuracy(%)")
    plt.plot(range(3, 16), accuracy)
    plt.show()
