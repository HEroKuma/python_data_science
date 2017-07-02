import random
from numpy import mat, shape

class BPNet(object):
    def __init__(self):
        self.eb = 0.01      # error tolerance
        self.iterator = 0
        self.eta = 0.1      # learning rate
        self.mc = 0.3       # optimization par
        self.maxiter = 2000
        self,nHidden = 4
        self.nOut = 1
        self.errlist = []
        self.dataMat = 0
        self.nSampNum = 0
        self.nSampDim = 0

    def logistic(self, net):

    def dlogit(self, net):

    def errorfunc(delf, inX):

    def normalize(self, dataMat):
        [m, n] = shape(dataMat)
        for i in range(n-1):
            dataMat[:, i] = (dataMat[:, i] - mean(dataMat[:, i])) / (std(dataMat[:, i]) + 1.0e - 10)
            return dataMat

    def loadDataSet(self, filename):
        self.dataMat = []; self.classLabels = []
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split()
            self.dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])
            self.classLabels.append(int(lineArr[2]))

        self.dataMat = mat(self.dataMat)
        m, n = shape(self.dataMat)
        self.nSampNum = m;
        self.nSampDim = n-1;

    def addcol(self, matrix1, matrix2):
        [m1, n1] = shape(matrix1)
        [m2, n2] = shape(matrix2)
        if m1 != m2:
            print("different rows, cannot merge")
            return

        mergMat = zeros((m1, n1+n2))
        mergMat[:, 0: n1] = matrix1[:, 0: n1]
        mergMat[:, n1: (n1+n2)] = matrix2[:, 0: n2]
        return mergMat

    def init_hiddenWB(self):
        self.hi_w = 2.0*(random.rand(self.nHidden, self.nSampDim - 0.5))
        self.hi_b = 2.0*(random.rand(self.nHidden, 1) - 0.5)
        self.hi_wb = mat(self.addcol(mat(self.hi_w), mat(self.hi_b)))

    def init_OutputWB(self):
        self.hi_w = 2.0 * (random.rand(self.nOut, self.nHidden) - 0.5)
        self.hi_b = 2.0 * (random.rand(self.nOut, 1) - 0.5)
        self.hi_wb = mat(self.addcol(mat(self.out_w), mat(self.out_b)))

    def bpTrain(self):

    def BPClassfier(self, start, end, steps=30):

    def classfyLine(delf, plt, x, z):

    def TrendLine(self, plt, color='r'):

    def drawClassScatter(self, plt):