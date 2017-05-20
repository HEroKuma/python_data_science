import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

#==========================================================================================================================
#execute 'matplotlib inline'
plt.rcParams['font.family']='SimHei'
df = pd.DataFrame(np.random.rand(6,4), index=['bar1','bar2','bar3','bar4','bar5','bar6'], columns=['A','B','C','D'])
print(df, '\n')

#describe
print('describe')
print(df.describe(), '\n')
#mean
print('mean')
print(df.mean(), '\n')
#max
print('max')
print(df.max(), '\n')
#min
print('min')
print(df.min(), '\n')
#median
print('median')
print(df.median(), '\n')
#quantile
print('quantile')
print(df.quantile(), '\n')
#sum
print('sum')
print(df.sum(), '\n')
#std
print('std')
print(df.std(), '\n')
#var
print('var')
print(df.var(), '\n')

from scipy import stats
x = [1.2, 1.15, 0.96, 0.9, 1.06]
print('gmean')
print(stats.gmean(x), '\n')

x = [300, 150]
print('hmean')
print(stats.hmean(x), '\n')

print('quantiles')
print(stats.mstats.mquantiles([1,2,3,4,5,6,7,8]), '\n')

print(df.cov, '\n')

plt.rcParams['axes.unicode_minus']=False
df.plot(kind='kde', title="PDF")
print('kurt')
print(df.kurt(), '\n')
print('skew')
print(df.skew(), '\n')

#==========================================================================================================================

import matplotlib.pyplot as plt
import numpy as np

# execute 'matplotlib inline'

x = np.random.randn(1000)
r = plt.boxplot(x, showfliers=True)
plt.show()
print(r['fliers'][0].get_data()[1], '\n')

from scipy.stats.mstats import mquantiles

print('四分位數', mquantiles(x), '\n')
IQR = mquantiles(x)[2] - mquantiles(x)[0]
print('IQR', IQR, '\n')
maximum = mquantiles(x)[2] + 1.5*IQR
print('最大值', maximum, '\n')
minimum = mquantiles(x)[2] - 1.5*IQR
print('最大值', minimum, '\n')

import pandas as pd
df = pd.DataFrame({'x':x})
df.plot(kind='kde', title='PDF')

print('平均數', x.mean(), '\n')
print('標準差', x.std(), '\n')
print('最大值', x.mean() + 3*x.std(), '\n')
print('最小值', x.min() - 3*x.std(), '\n')
print('>max之異常值', df[df['x'] > x.mean() + 3*x.std()], '\n')
print('<max之異常值', df[df['x'] < x.mean() - 3*x.std()], '\n')

#==========================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#run 'matplotlib inline'

plt.rcParams['font.family']='SimHei'

df = pd.read_csv('A_LVR_LAND_A.CSV', encoding='big5hkscs')
#print(df[:10], '\n')
print(df.corr(), '\n')
plt.rcParams['axes.unicode_minus']=False
df.plot(kind='scatter', title='散佈圖(高相關)', figsize=(6,4), x='總價元', y='建物移轉總面積平方公尺', marker='+')
df.plot(kind='scatter', title='散佈圖(中相關)', figsize=(6,4), x='建物現況格局-房', y='建物現況格局-衛', marker='+')
df.plot(kind='scatter', title='散佈圖(低相關)', figsize=(6,4), x='建物現況格局-房', y='總價元', marker='+')

#==========================================================================================================================
from subprocess import call

f=open('tesco.csv', 'r')
print(f.read(), '\n')
call(['python3', 'apriori.py -f tesco.csv -s 0.3 -c 0.5'])

#==========================================================================================================================
from datetime import datetime
from datetime import timedelta

now = datetime.now()
print(now, '\n')
print(now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, '\n')
print(now.weekday(), '\n')
print(now.isoformat(), '\n')

delta = datetime(2016, 11, 27, 16, 39, 47, 748848) - datetime(2015, 1, 1)
print(delta, '\n')
print(delta.days, delta.seconds, delta.microseconds, '\n')

print('timedelta')
print(datetime(2015, 10, 10, 8, 15) + timedelta(12, 10, 10), '\n')

#parse
date = datetime.strptime('Jun 1 2005 1:33PM', '%b %d %Y %I:%M%p')
print(date, '\n')

date = datetime.strptime('2011/02/10 13:33', '%Y/%m/%d %H:%M')
print(date, '\n')

date = datetime.strptime('2016-05-30 13:33:50', '%Y-%m-%d %H:%M:%S')
print(date, '\n')

date = datetime.strptime('2016-Aug-30 13:33:50', '%Y-%b-%d %H:%M:%S')
print(date, '\n')

print(datetime.strftime(date, '%Y-%m-%d %H:%M:%S'), '\n')

print(datetime.strftime(date, '%Y-%m-%d %H:%M:%S'), '\n')

#==========================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./globall_land_temperature/GlobalTemperatures.csv", encoding='utf8')
#print(df, '\n')
df = df[df['LandAverageTemperature'].notnull()]
df = df.set_index(df['dt'], drop=True)
del df['dt']
print(df.head(), '\n')
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
print(df.index.weekday,'\n')

plt.style.use('ggplot')
#df.plot(kind='line', figsize=(20, 10))
print('df', '\n')
df2 = df.groupby(df.index.year).mean()
df2.set_xlim(1750,2010)
df2['LandAverageTemperature'].plot(kind='line', figsize=(20, 10), color='r')

print('min', '\n')
df_min = df.groupby(df.index.year).min()
df_min.set_xlim(1750,2010)
df_min['LandAverageTemperature'].plot(kind='line', figsize=(20, 10), color='g')

print('max', '\n')
df_max = df.groupby(df.index.year).max()
df_max.set_xlim(1750,2010)
df_max['LandAverageTemperature'].plot(kind='line', figsize=(20, 10), color='b')

print('Q', '\n')
df_Q = df.resample('Q-NOV').mean()
df_Q[df_Q.index.year > 2012]
df_Q.set_xlim(1750,2010)
df_Q['LandAverageTemperature'][df_Q.index.year > 2012].plot(kind='line', style='-o', figsize=(20, 10), color='r')

print('M', '\n')
df_M = df.resample('M').mean()
df_M.set_xlim(1750,2010)
df_M['LandAverageTemperature'][df_M.index.year > 2014].plot(kind='line', style='-o', figsize=(20, 10), color='r')

#==========================================================================================================================
import pandas as pd
import matplotlib.pyplot as plot

plt.style.use('ggplot')

plt.rcParams['font.family']='SimHei'
df = pd.DataFrame({'BookId':['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10'],'Income':[1000, 3000, 30000, 24000, 300, 400, 800, 2300, 12000, 1800]})
df = df.sort_values(by=['Income'], ascending=False).reset_index(drop=True)
#print(df, '\n')

p = []
for i in range(len(df)):
    x = df['Income'][i]/sum(df['Income'])
    if(i!=0):
        p.append(x+p[i-1])
    else:
        p.append(x)

print(p, '\n')

fig, ax1 = plt.subplots()
ax1.bar(df.index, df['Income'], align='center')
ax1.set_ylabel('收入')
ax1.set_xlabel('圖書ID')

ax2 = ax1.twinx()
ax2.plot(df.index, p, 'r-')
ax2.set_ylabel('收入累積比例')
ax2.grid(False)

for i in range(len(p)):
    if(p[i] > 0.8):
        ax2.annotate(round(p[i], 3), xy=(df.index[i], p[i]), xytext=(df.index[i], p[i]+0.08), arrowprops=dict(facecolor='black'))
        break

plt.xticks(df.index, df['BookID'])
plt.title('analysis')

plt.show()