# -*- coding: utf-8 -*-
import pandas as pd

#Grouping
df = pd.DataFrame({'key1':['A','B','C','B','B','A','D','E'], 'key2':['A','B','D','B','D','A','C','D'],
'count1':[10,20,23,42,51,76,54,80], 'count2':[1.2,3.2,5.2,1.3,6.4,3.6,8.5,9.2]})
print('DataFrame')
print(df,'\n\n\n')

print('mean of key1')
print(df.groupby('key1').mean(), '\n\n\n')

print('sum of key1')
print(df.groupby('key1').sum(), '\n\n\n')

print('sum of key1 & key2')
print(df.groupby(['key1','key2']).sum(), '\n\n\n')

print('size of key1 & key2')
print(df.groupby(['key1','key2']).size(), '\n\n\n')

print('sum of key1 & key2 & just show count1')
print(df.groupby(['key1','key2'])['count1'].sum(), '\n\n\n')

#===================================================================================================================
df = pd.read_csv("NHI.csv",encoding='big5hkscs')
#print(df,'\n')
print('Label')
print(df.pivot_table(values = '腸病毒健保就診人次', index = ['年', '就診類別'], columns = '縣市', aggfunc='sum'),'\n\n\n')

#===================================================================================================================
#visualization
import matplotlib.pyplot as plt
import numpy as np

#To see all plot style: plt.style.available
#直方 with ggplot
plt.style.use('ggplot')
x = np.random.normal(loc=60, scale=15, size=100)  # loc:平均數, scale:標準差, size:資料數
#print(x,'\n')
plt.hist(x, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.show()
print('\n\n\n')

#長條 with dark_background
plt.style.use('dark_background')
x = [1,2,3,4,5]
y = [123,456,78,823,4]
plt.bar(x, y, align='center')
plt.show()
print('\n\n\n')

#折線 with grayscale
plt.style.use('grayscale')
plt.plot(x,y)
plt.show()
print('\n\n\n')

#散佈圖 with classic
plt.style.use('classic')
x = [30, 43, 83, 23, 42, 47, 92, 91, 93]
y = [123, 645, 54, 54, 547, 89, 75, 976, 361]
plt.scatter(x,y)
plt.show()
print('\n\n\n')

#箱型圖
x = np.random.randn(1000)
#print(x)
print(x.max(), x.min())
plt.boxplot(x, showfliers=False)
plt.show()
print('\n\n\n')

#圓餅圖
labels = ['A','B','C','D','E','F','G']
y = [123,55,38,234,356,8,6]
plt.figure(figsize=(4,4))
plt.pie(y, labels=labels)
plt.show()
print('\n\n\n')

#===================================================================================================================
#advanced
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

#折線圖
fig = plt.figure(figsize=(4,3))

x1 = [1,2,3,4,5]
y1 = [123,45,56,4,234]
x2 = [1,2,3,4,5]
y2 = [423,678,5,34,37]

plt.plot(x1, y1, '-')
plt.plot(x2, y2, '-')
plt.show()
print('\n\n\n')

#長條圖
fig = plt.figure(figsize = (4,3))

x1 = [1,2,3,4,5]
y1 = [123,45,56,4,234]
x2 = [1,2,3,4,5]
y2 = [423,678,5,34,37]

plt.bar(np.array(x1)-0.2, y1, color='g', width=0.4, align='center')
plt.bar(np.array(x2)+0.2, y2, color='b', width=0.4, align='center')
plt.show()
print('\n\n\n')

#===================================================================================================================
#axis
import matplotlib.pyplot as plt

plt.style.use('ggplot')
fig = plt.figure(figsize=(4,3))
x = [1,2,3,4,5]
y = [123,45,56,4,234]

plt.rcParams['font.family']='SimHei'  # to show chinese
#Find more font style: [f.name for f in matplotlib.font_manager.fontManager.ttflist]

plt.bar(x, y, align='center')
plt.title('中文標題')
plt.xlabel('X軸')
plt.ylabel('y軸')
plt.xticks(x, ['刻度A','刻度B','刻度C','刻度D','刻度E'])
plt.show()
print('\n\n\n')

#two font style
fig = plt.figure(figsize=(4,3))
x = [1,2,3,4,5]
y = [123,45,56,4,234]

font1 = {'fontname': 'Times New Roman'}
font2 = {'fontname': 'Arial'}

plt.bar(x, y, align='center')
plt.title('Title')
plt.xlabel('X axis', **font1)
plt.ylabel('y axis', **font2)
plt.xticks(x, ['tick A','tick B','tick C','tick D','tick E'], **font2)
plt.show()
print('\n\n\n')

#===================================================================================================================
#combine plot
import matplotlib.pyplot as plt

plt.style.use('ggplot')
x = [1,2,3,4,5]
y = [123,45,56,4,234]

ax1 = plt.subplot(211)
plt.plot(x,y,'-')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplot(212, sharex=ax1)
plt.bar(x,y,align='center')
plt.show()
print('\n\n\n')

plt.figure(figsize=(5,3))
plt.plot(x,y,'-')
plt.bar(x,y,align='center',color='b')
plt.show()
print('\n\n\n')

#===================================================================================================================
#plot style
import matplotlib.pyplot as plt

plt.style.use('ggplot')
fig = plt.figure(figsize=(4,3))
x = [1,2,3,4,5]
y = [123,45,56,4,234]
plt.plot(x, y, '-')
plt.show()
print('\n\n\n')

fig = plt.figure(figsize=(4,3))
plt.plot(x,y,'b-.')
plt.show()
print('\n\n\n')

fig = plt.figure(figsize=(4,3))
plt.plot(x,y,'b-o')
plt.show()
print('\n\n\n')

fig = plt.figure(figsize=(4,3))
plt.plot(x,y,'*')
plt.show()
print('\n\n\n')
#about matplotlib: http://matplotlib.org/api/pyplot_api.html

#===================================================================================================================
#pandas visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from subprocess import call

#run '%matplotlib inline' in shell first
plt.style.use('ggplot')
plt.rcParams['font.family']='SimHei'
df = pd.DataFrame(np.random.rand(6,4), index=['bar1','bar2','bar3','bar4','bar5','bar6'], columns=['A','B','C','D'])
#print(df)
df.plot(kind='bar', title='長條圖', figsize=(6,4))
df.plot(kind='bar', title='堆疊長條圖', figsize=(6,4), stacked=True)
df.plot(kind='area', title='堆疊面積圖', figsize=(6,4))
df.plot(kind='area', title='堆疊面積圖', figsize=(6,4), stacked=False)
df.plot(kind='line',style='--', title='折線圖', figsize=(6,4))
df['A'].plot(kind='pie', title='圓餅圖', figsize=(4,4))
df.plot(kind='line', title='折線圖', figsize=(8,6), subplots=True, style={'A':'-', 'B':'--', 'C':'-o', 'D':'-.'})
ax = df.plot(kind='bar', figsize=(6,4), stacked=True)
df.plot(kind='line', style='--', figsize=(6,4), ax=ax)
print('\n\n\n')


df2 = pd.DataFrame({'score':np.random.normal(70, 10, 200)})
df2.plot(kind='hist', title='直方圖', figsize=(6,4))
df2.plot(kind='kde', title='機率密度圖')
print('\n\n\n')


df3 = pd.DataFrame({'A': np.random.randn(1000), 'B': np.random.randn(1000)})
df3.plot(kind='hexbin', title='蜂窩圖', x='A', y='B', gridsize=10)
print('\n\n\n')

df4 = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100), 'C': np.random.randn(100)})
df4.plot(kind='scatter', title='散佈圖', figsize=(6,4), x='A', y='B')
df4.plot(kind='scatter', title='散佈圖', figsize=(6,4), x='A', y='B', c='C')
print('\n\n\n')
#http://pandas.pydata.org/pandas-docs/stable/