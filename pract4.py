import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df =pd.read_csv('BHP.csv')
df.shape
df.describe()
df.isnull().sum()
df.dropna(inplace=True)
df.head()
df['size'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

def convertrange(x):   
    temp =x.split("-")
    if len(temp)==2:
        return(float(temp[0])+float(1))/2
    try:
        return float(x)
    except:
        return None

df['total_sqft']=df['total_sqft'].apply(convertrange)
df['location']=df['location'].apply(lambda x: x.strip())

df['persq']= df['price']/df['total_sqft']
df.dropna(inplace=True)
df


from sklearn.preprocessing import StandardScaler
scale = df['total_sqft'].values.reshape(-1, 1)

scaler = StandardScaler()
model = scaler.fit_transform(scale)
scaled_data = model.transform(scale)


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
df2=df
x1= df.drop["price"]
x= x1.drop['Location']
y= df['price']

x = std.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

mse_train = mean_squared_error(y_train,y_train_pred)
mse_test = mean_squared_error(y_test,y_test_pred)

r2_train = r2_score(y_train,y_train_pred)
r2_test = r2_score(y_test,y_test_pred)

print("Training set Mean Squared Error: {:.2f}".format(mse_train))
print("Testing set Mean Squared Error: {:.2f}".format(mse_test))
print("Training set Accuracy: {:.2f}".format(r2_train))
print("Testing set Accuracy: {:.2f}".format(r2_test))


lr.coef_
lr.intercept_
lr.predict([[2,1056,2.0,3699.8106]])