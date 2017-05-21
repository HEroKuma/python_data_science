import pandas as pd
import numpy as np

df1 = pd.DataFrame({'item':['i01','i02','i03','i04','i05','i07'], 'cid':['c01','c02','c03','c04','c05','c07']})
df2 = pd.DataFrame({'cid':['c01','c02','c03','c04','c05','c06'], 'num_products':['2','13','12','4','5','6']})
print(df1,'\n')
print(df2,'\n')

#inner join
print("Inner Join")
print(pd.merge(df1,df2, left_on='cid', right_on='cid', how='inner'),'\n')

#outer join
print("Outer Join")
print(pd.merge(df1,df2, left_on='cid', right_on='cid', how='outer'),'\n')

#left join
print("Left Join")
print(pd.merge(df1,df2, left_on='cid', right_on='cid', how='left'),'\n')

#right join
print("Right Join")
print(pd.merge(df1,df2, left_on='cid', right_on='cid', how='right'),'\n')

#diff
def diff_dataframe(df1, df2):
    for i in range(len(df1.index)):
        for j in range(len(df2.index)):
            if(np.array_equal(df1[i:i+1].values, df2[j:j+1].values)):
                df1 = df1.drop(i).reset_index(drop=True)
    return df1

print("Diff Join")
print(diff_dataframe(df1, df2),'\n')

#======================================================================================================================
df = pd.DataFrame({'col1':['a','a','a','b','b','c','d','e'], 'col2':[1,1,2,2,3,3,4,5]})
print(df,'\n')

print("Duplicated")
print(df.duplicated(),'\n')

print("Duplicated")
print(df.duplicated(['col1']),'\n')

print("Drop_duplicates")
print(df.drop_duplicates(),'\n')

print("Drop_duplicates")
print(df.drop_duplicates(['col1']),'\n')

print("Drop_duplicates")
print(df.drop_duplicates(['col1']).reset_index(drop=True),'\n')

print("Drop_duplicates")
print(df.drop_duplicates(['col1'], keep='last'),'\n')

#======================================================================================================================
df = pd.DataFrame({'sex':['male','male','female','male','female','female']})
print(df,'\n')

sex_to_boolean = {'female':0, 'male':1}
df['code'] = df['sex'].map(sex_to_boolean)
print(df,'\n')

#======================================================================================================================

df = pd.DataFrame({'col1':['c01','c02','c03','c04','c05'], 'col2':[54,'NULL','NaN','NaN',78], 'col3':[321,34,'NULL','NaN',34]})
print(df,'\n')

print('replace','\n')
print(df['col2'].replace('NaN',0),'\n')
print(df.replace('NaN',0),'\n')
print(df.replace({'NaN':0,'NULL':-1}),'\n')

#======================================================================================================================

df =pd.DataFrame({'id':['a01','a02','a03','a04','a05'], 'score':[74,59,98,100,60]})
print(df,'\n')

bins = [0,60,70,80,90,101]
print(pd.cut(df['score'], bins),'\n')

print(pd.cut(df['score'], bins, right=False),'\n')

labels = ['F','D','C','B','A']
print(pd.cut(df['score'], bins, right=False, labels=labels),'\n')

df['label1'] = pd.cut(df['score'], bins, right=False, labels=labels)
print(df,'\n')

#======================================================================================================================
df = pd.DataFrame({'col1':['a',float('NaN'), 'a', None, 'b', 'c', 'd', 'e'], 'col2':[1,1,3,2,float('NaN'),3,4,5]})
print(df)

print(df['col1'].isnull(), '\n')
print(df['col2'].isnull(), '\n')