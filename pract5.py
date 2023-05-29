import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score

df = pd.read_csv('Social_Network_Ads.csv')
df.head()

df.isnull().sum()
df1.head()
df1.head()
le = LabelEncoder()
df1['Gender'] = le.fit_transform(df1['Gender'])

std = StandardScaler()
x = df1.drop(['Purchased'], axis=1)
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)


model = LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(classification_report(y_test,y_pred))
cor = confusion_matrix(y_test,y_pred)
cor

cmd = ConfusionMatrixDisplay(cor).plot()
cmd
acc =accuracy_score(y_test,y_pred)
acc
pc = precision_score(y_test, y_pred)
pc
rc = recall_score(y_test,y_pred)