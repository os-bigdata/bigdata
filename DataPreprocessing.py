import pandas as pd
import numpy as np

d = pd.read_csv(r"C:\Users\sonas\Desktop\housing.csv")
d.describe()

print(d.shape,d.size,d.ndim)

d.isnull().sum()

d.columns

d.columns.size

d = d.replace(0,np.NaN)
d.isnull().sum()

d.head(10)

d.dropna(thresh = 20)

d.isnull().sum()

d = d.fillna(value=d.loc[:,d.columns].mean())
d.isnull().sum()

d= d.dropna()
d.isnull().sum()

d.dtypes.value_counts()

print(d.shape,d.size,d.ndim)

x = d.iloc[:,:-1]
y = d.iloc[:,-1]
x

y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size=0.2)
print("Train Size: ",(len(x_train)/len(x))*100)
print("Test Size: ",(len(x_test)/len(x))*100)
