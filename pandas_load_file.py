#===================================================Load csv or excel============================================
'''
import pandas as pd
import codecs

df = ''
with codecs.open('A_LVR_LAND_A.CSV', 'r', 'Big5', 'ignore') as file:
    df = pd.read_table(file, delimiter=',')
print(df[:5])

df2 = pd.read_excel('Sample.xlsx', sheetname = 'sheet1')
'''

#===================================================Load MySql===================================================
'''
import mysql.connector
import pandas.io.sql as sql

#Connect setting
config = {
    'user':'root',
    'password': 'your password',
    'host': '127.0.0.1'
    'database': 'PDS'
}
#Connect to DB
cnx = mysql.connector.connect(**config)

df = sql.read_sql('select * from xxxxxx;', cnx)
df
'''

#===================================================Load Json===================================================
'''
import json
import requests

res = requests.get("http://cloud.culture.tw/frontsite/trans/SearchShowAction.do?method=doFindTypeJ&category=6")
print(res.text)
result = json.loads(res.text)
result[:10]
df2 = pd.DataFrame(result)  # Load Json to DF
print(df2)
#DataFrame.append(data)  #append data
'''

#===================================================Load XML====================================================
'''
import xml.etree.ElementTree as et
import requests
res = requests.get("https://cloud.culture.tw/frontsite/trans/emapOpenDataAction.do?method=exportEmapXMLByMainType&mainType=10")
#print(res.text)
#et.ElementTree(XMLFile)  # XML->String XMLFile
tree = et.ElementTree(et.fromstring(res.text))
root = tree.getroot()
print(root.tag)  # saw the content in tag
print(root.attrib)
for elem in tree.iter():  # scan all tag content in xml tree
    print(elem.tag, elem.attrib)

info = []
for elem in tree.iter(tag='Info'):  # for special tag
    print(elem.tag, elem.attrib)
    info.append(elem.attrib)
'''