import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

d = pd.read_csv(r"C:\Users\sonas\Desktop\sala.csv")
x = d.iloc[:,:-1].values
y = d.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=1)

reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

plt.scatter(x_train,y_train, color="blue")
plt.plot(x_train,reg.predict(x_train), color = "green")
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title("Salary vs Experience")
plt.show()

from numpy import cov
from scipy.stats import pearsonr
co = cov(y_test,y_pred)
corr = pearsonr(y_test,y_pred)
print("Covariance: ",co)
print("Correlation: ",corr)
