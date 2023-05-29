import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv('Academic_Performance.csv')
df.drop(['STUDENT_ID'],axis=1, inplace=True)
df.drop(['STUDENT_ID'],axis=1, inplace=True)
df.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohm = OneHotEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='constant', fill_value='missing') 
newdf.GENDER=imputer.fit_transform(newdf["GENDER"].values.reshape([-1,1]))[:,0]

ohm = pd.get_dummies(df,columns=['PLACEMENT','EDUCATION_TYPE'])

cat = df['Gender'].replace(['M','F'],[2,1])