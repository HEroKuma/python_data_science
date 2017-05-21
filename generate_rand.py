import numpy as np

data = np.random.rand(500,2)*500

f = open('testSet2.txt', 'w')
for i, j in data:
    #print(i,' ', j)
    val = str(i) + ' ' + str(j) + '\n'
    f.write(val)

f.close()