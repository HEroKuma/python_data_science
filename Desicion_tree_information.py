import pandas as pd

df = pd.read_csv('iris.csv', encoding='big5hkscs')

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=5)
tree.fit(df[['花萼長度','花萼寬度','花瓣長度','花瓣寬度']], df[['屬種']])

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot', feature_names=['花萼長度','花萼寬度','花瓣長度','花瓣寬度'], class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginics'])

tree.feature_importances_.tolist()
df = pd.DataFrame({'feature': ['花萼長度','花萼寬度','花瓣長度','花瓣寬度'], 'feature_importance': tree.feature_importances_.tolist()})
df = df.sort_values(by=['feature_importance'], ascending=False).reset_index(drop=True)
print(df, '\n')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.02*height, '%f'%float(height), ha='center', va='bottom')

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure(figsize=(8, 4))

plt.rcParams['font.family']

gini = plt.bar(df.index, df['feature_importance'], align='center')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.xticks(df.index, df['feature'])

autolabel(gini)
plt.show()

#====================================================================================================================
#Entropy
import pandas as pd

df = pd.read_csv('iris.csv', encoding='big5hkscs')

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree.fit(df[['花萼長度','花萼寬度','花瓣長度','花瓣寬度']], df[['屬種']])

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot', feature_names=['花萼長度','花萼寬度','花瓣長度','花瓣寬度'], class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginics'])

import matplotlib.image as mping
import matplotlib.pyplot as plt

img = mping.imread('tree_en.png')
fig = plt.figure(figsize=(10, 8))
plt.imshow(img)

tree.feature_importances_.tolist()
df = pd.DataFrame({'feature': ['花萼長度','花萼寬度','花瓣長度','花瓣寬度'], 'feature_importance': tree.feature_importances_.tolist()})
df = df.sort_values(by=['feature_importance'], ascending=False).reset_index(drop=True)

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure(figsize=(8, 4))

plt.rcParams['font.family']

gini = plt.bar(df.index, df['feature_importance'], align='center')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.xticks(df.index, df['feature'])

autolabel(gini)
plt.show()