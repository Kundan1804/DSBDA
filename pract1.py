import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("dirtydata.csv")
df.head()
df.describe()
df.isnull().sum()
df.shape
df['Date'] = pd.to_datetime(df['Date'])
df.head()
df['Calories'] = df['Calories'].fillna(df['Calories'].mean())

cat=pd.cut(df.Age,bins=[19,25,30,35,45],labels=['A','B','C','D'])
df.insert(5,'Age_Grp',cat)
new.duplicated().sum()
new.drop_duplicates(inplace=True)
df['Pos']=df['Position'].replace(['SG','PF','PG','SF','C'],[1,2,3,4,5])  #categorical to quantative 