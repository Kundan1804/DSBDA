import pandas as pd
import numpy as np
import statistics
import math
df=pd.read_csv("Age-Income-Dataset.csv")
df.head()
avg = sum(df['Income'])/len(df['Income'])
avg
df['Income'].mean()
incnp = np.array(df['Income'])
incpd = pd.Series(df['Income'])
npmean = np.mean(incnp)
pdmean = incpd.mean()

M=sorted(df['Income'])
n = len(M)
if n%2 == 0:
    index=(n//2)
    median=(M[index]+M[(index-1)])/2
else:
    median = M[n-1]/2
print(median)

md = statistics.median(df['Income'])
md
freq={}
for value in M:
    if value in freq:
        freq[value]+=1
    else:
        freq[value]=1

max_frq=max(freq.values())
for value, key in freq.items():
    if key == max(freq):
        print(freq[value])

df.groupby("Income").count()

n = len(df['Income'])
avg = df['Income'].mean()
variance = sum((item - avg)**2 for item in df['Income'])/(n-1)
variance

incpd.skew()
def minimum(income): 
    current_min = income[0] 
    for num in income:
        if num < current_min:
            current_min = num
            return current_min 
print(minimum(df['Income']))

#using formula
max_value = None 
for num in income:
    if (max_value is None or num > max_value): 
        max_value = num
print('Maximum value:', max_value)