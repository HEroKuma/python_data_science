from numpy import *
import time
import matplotlib.pyplot as plt

def Distance(vector1, vector2):  # Euclidean distance in this func
    return np.sqrt(np.sum(np.asarray(vector1 - vector2)**2))

def initCentroids(dataSet, k):  # In random samples
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    print(centroids)
    return centroids

def kmeans(dataSet, k):  # Cluster
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    centroids = initCentroids(dataSet, k)
    count = 1
    while clusterChanged:
        print(count)
        count = count+1
        clusterChanged = False
        for i in range(numSamples):
            minDist = 10000.0
            minIndex = 0
            for j in range(k):
                distance = Distance(centroids[j, :], dataSet[i, :])  #Caculate the distance between 2 vectors
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2

        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis = 0)

    print('Complete!\n')
    return centroids, clusterAssment

def showCluster(DataSource, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    print(centroids)
    plt.figure(figsize=(12,9))
    if dim != 2:
        print('Can\'t draw with more than dim()=2')
        return 1
    
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print('k is too large')
        return 1

    for i in range(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^r', '+r', 'sb', 'db', '<b', 'pb']
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

    plt.show()

## step 1: load data
dataSet = []
fileIn = open('./testSet2.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split(' ')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: clustering...
dataSet = mat(dataSet)
k = 3
centroids, clusterAssment = kmeans(dataSet, k)
print(centroids)
## step 3: show the result
showCluster(dataSet, k, centroids, clusterAssment)