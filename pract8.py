import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('tested.csv')
df.head()
df.describe()
df.isnull().sum()
df.isnull().sum()
df.drop(['Cabin'],axis=1, inplace = True)
df.dropna(subset = ['Age','Fare'], inplace=True)
sns.boxplot(data = df, x=df['Fare'])
outier = []
upr =0 
df["Fare"].head

outier = []
upr =0 
df["Fare"].head

df = df[(df['Fare']<upr) & (df['Fare']>lwr)]
df['Fare']

sns.scatterplot(x='age',y='fare',data=x1, hue='sex')
sns.displot(x1['age'],kde=True, aspect=4, color='green')
sns.displot(data=x1,x='age', hue='sex',col='pclass',kde=True) 
sns.displot(data=x1,x='embarked', hue='alive',col='pclass',kde=True)
sns.barplot(x='sex',y='age', data=x1)
sns.countplot(x='pclass',hue='sex',data=x1)
sns.displot(x1['fare'],kde=True, aspect=4)          #only for continous data
   