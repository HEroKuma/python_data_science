#相關性分析
'mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']='SimHei'

#Load csv
df = pd.read_csv('A_LVR_LAND_A.CSV', encoding='big5hkscs')
#print(df.corr(), '\n')
plt.rcParams['axes.unicode_minus']=False
#df.plot(kind='scatter', title='散佈圖', figsize=(18, 12), x='建物移轉總面積平方公尺', y='總價元', marker='+')

#資料預處理
df = df[df['建物移轉總面積平方公尺'] > 0]
df = df[df['建物移轉總面積平方公尺'] < 1000]  # 刪除極端值
df['建物移轉總面積平方公尺'] = df['建物移轉總面積平方公尺']/10000
df = df[df['總價元'] > 0]
df['總價元'] = df['總價元']/10000
df = df[df['總價元'] < 20000]  # 刪除極端值
#print(df['總價元'], '\n')
df.plot(kind='scatter', title='散佈圖2', figsize=(18, 12), x='建物移轉總面積平方公尺', y='總價元', marker='+')

#訓練資料集切分
from sklearn.cross_validation import train_test_split
x = df[['建物移轉總面積平方公尺']]
y = df[['總價元']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
print(x_train.head(), '\n')

#==================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
plt.style.use('ggplot')

#Linear regression 物件
regr = linear_model.LinearRegression()

#訓練模型
regr.fit(x_train, y_train)

print('parameter: \n', regr.coef_)
print('mean squared error, MSE: %.2f: \n' % np.mean((regr.predict(x_test) - y_test) **2))

plt.scatter(x_test, y_test, color='b', marker = 'x')
plt.plot(x_test, regr.predict(x_test), color='green', linewidth=1)

plt.ylabel('總價元')
plt.xlabel('建物移轉總面積平方公尺')

plt.show()

#==================================================================================================================
#多變項線性迴歸
df['建物現況格局-衛'] = df['建物現況格局-衛']/10000
x = df[['建物移轉總面積平方公尺','建物現況格局-衛']]
y = df[['總價元']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
print(x_train.head(), '\n')

#標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_nor = sc.transform(x_train)
x_test_nor = sc.transform(x_test)
#print(x_train_nor, '\n')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

#Linear regression 物件
regr = linear_model.LinearRegression()

#訓練模型
regr.fit(x_train_nor, y_train)

print('Normalized: \n', regr.coef_)
print('MSE: %.2f' % np.mean((regr.predict(x_test_nor) - y_test) **2))


#==================================================================================================================
#多項式非線性迴歸
from sklearn.cross_validation import train_test_split

x = df[['建物移轉總面積平方公尺']]
y = df[['總價元']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#創造高維變項
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures

quadratic = PolynomialFeatures(degree=2)
x_train_raw = x_train.copy()
x_test_raw = x_test.copy()
x_train = quadratic.fit_transform(x_train_raw)
x_test = quadratic.fit_transform(x_test_raw)

x_fit = pd.DataFrame(np.arange(0, 0.1, 0.001))

#Linear regression 物件
regr = linear_model.LinearRegression()

#訓練模型
regr.fit(x_train, y_train)

print('parameter: \n', regr.coef_)
print('MSE: %.2f: \n' % np.mean((regr.predict(x_test) - y_test) **2))

plt.figure(figsize=(18, 12))
plt.scatter(x_test_raw, y_test, color='b', marker = 'x')
plt.plot(x_fit, regr.predict(quadratic.fit_transform(x_fit)), color='green', linewidth=1)
plt.show()

#==================================================================================================================
#logistic regression
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

plt.style.use('ggplot')

z = np.arange(-8, 8, 0.1)
y = sigmoid(z)
plt.plot(z, y)
plt.show()

import pandas as pd

df = pd.read_csv('iris.csv', encoding='big5hkscs')
print(df.head(), '\n')

from sklearn.cross_validation import train_test_split

x = df[['花萼長度','花萼寬度']]
y = df['屬種']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
print(x_train.head(), '\n')

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train)
x_train_nor = sc.transform(x_train)
x_test_nor = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
import matplotlib.patches as mpatches

lr = LogisticRegression()
lr.fit(x_train_nor, y_train)
pred_y = lr.predict(x_train_nor)

for i in range(len(x_train_nor)):
    if(lr.predict(x_train_nor[i].reshape(1, -1)) == 'Iris-setosa'):
        plt.plot(x_train['花萼長度'].reset_index(drop=True)[i], x_train['花萼寬度'].reset_index(drop=True)[i], 'b.')
    elif(lr.predict(x_train_nor[i].reshape(1, -1)) == 'Iris-versicolor'):
        plt.plot(x_train['花萼長度'].reset_index(drop=True)[i], x_train['花萼寬度'].reset_index(drop=True)[i], 'gx')
    else:
        plt.plot(x_train['花萼長度'].reset_index(drop=True)[i], x_train['花萼寬度'].reset_index(drop=True)[i], 'r+')

blue_patch = mpatches.Patch(color='blue', label='Iris-setosa')
green_patch = mpatches.Patch(color='green', label='Iris-versicolor')
red_patch = mpatches.Patch(color='red', label='Iris-virginica')
plt.show()

for i in range(len(x_train_nor)):
    if(lr.predict(x_train_nor[i].reshape(1, -1)) == 'Iris-setosa'):
        plt.plot(x_train['花萼長度'].reset_index(drop=True)[i], x_test['花萼寬度'].reset_index(drop=True)[i], 'b.')
    elif(lr.predict(x_train_nor[i].reshape(1, -1)) == 'Iris-versicolor'):
        plt.plot(x_train['花萼長度'].reset_index(drop=True)[i], x_test['花萼寬度'].reset_index(drop=True)[i], 'gx')
    else:
        plt.plot(x_train['花萼長度'].reset_index(drop=True)[i], x_test['花萼寬度'].reset_index(drop=True)[i], 'r+')

blue_patch = mpatches.Patch(color='blue', label='Iris-setosa')
green_patch = mpatches.Patch(color='green', label='Iris-versicolor')
red_patch = mpatches.Patch(color='red', label='Iris-virginica')
plt.show()

#==================================================================================================================
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment = 'center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import numpy as np

np.round(lr.predict_proba(x_test_nor), 3)

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test, lr.predict(x_test_nor))
print(cnf_matrix)

import itertools

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix, without normalization')

plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, lr.predict(x_test_nor), target_names=target_names))