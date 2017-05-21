import pandas as pd

df = pd.read_csv('StudentJob.csv', encoding='big5')

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
df.salary = df.salary.astype(float)
df.working = df.working.astype(float)
x = df[['salary', 'working']].values

km = KMeans(n_clusters=3)
y_pred = km.fit_predict(x)
plt.figure(figsize=(10, 6))
plt.xlabel('Salary')
plt.ylabel('Rate of working')
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

print(km.cluster_centers_, '\n')