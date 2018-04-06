from numpy import *
import operator

def createDataSet():
    group = array([[1,1],[2,1],[1,3],[3,4],[4,3]])
    labels = ['A','A','A','B','B']
    return group,labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def classify(inX,dataSet,labels,k):
    # 计算距离
    dataSetSize = dataSet.shape[0]  # 读取数组第一维度的长度
    diffMat = tile(inX,(dataSetSize,1))-dataSet  # 建立一个dataSetSize×len(inX)的0矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 对每一行的数据求和
    distances = sqDistances ** 0.5
    # 按离该点的距离按从近到远的距离排序
    sortedDistIndicies = distances.argsort()  #

    # 计算前k个点的分类，并返回最多点的分类
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10  # 测试集占总样本的比例
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) # 测试集的样本数
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],
                                    datingLabels[numTestVecs:m],3)
        print("The classifier cam back with ：{}, the real answer is :{}."
              .format(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount += 1.0
    print("The total error rate is : {}"
          .format(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input(\
        "percentage of time spend playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify((inArr-\
                                    minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: {}"
          .format(resultList[classifierResult-1]))

classifyPerson()

