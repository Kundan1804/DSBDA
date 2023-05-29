import seaborn as sns
df=sns.load_dataset('iris')
df.describe()
df.isnull().sum()
df.dtypes
df.nunique()
sns.displot(df['sepal_length'],kde=True, aspect=4)
sns.displot(df['sepal_width'],kde=True, aspect=4)
sns.scatterplot(x='species',y='sepal_width',data=df)
sns.boxplot(x='species',y='sepal_width',data=df)
sns.swarmplot(x='species',y='petal_width',dodge=True, size=3,data=df)
sns.lineplot(x='sepal_length',y='petal_length', data=df)  #shadow is the confidence interval i.e. the dataset contains multiple y values for each x value
sns.lineplot(x='sepal_length',y='sepal_width', data=df)
sns.lineplot(x='petal_length',y='petal_width', data=df, ci=None)