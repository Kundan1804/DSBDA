import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('Iris.csv')
df.head()

df.describe()
df.isnull().sum()
df.dtypes
df.nunique()

df.head()

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

x = df.drop(['Species'], axis=1)
y = df['Species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

std = StandardScaler(with_mean=False)
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB
gn = GaussianNB()

model = gn.fit(x_train,y_train)
y_pred = model.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score, classification_report
acc= accuracy_score(y_test,y_pred)
acc

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
conf = confusion_matrix(y_test,y_pred)
print(conf)

M = ConfusionMatrixDisplay(conf).plot()
M

TN = conf[0, 0]
FN = conf[1, 0] + conf[2, 0]
FP = conf[0, 1] + conf[0, 2]
TP = conf[1, 1] + conf[1, 2] + conf[2, 1] + conf[2, 2]
print("TN", TN)
print("FN", FN)
print("FP", FP)
print("TP", TP)

