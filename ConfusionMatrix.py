import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d= pd.read_csv(r"C:\Users\sonas\Downloads\Social_Network_Ads.csv")
x= d.iloc[:,:-1]
y = d.iloc[:,-1]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
sv = SVC(kernel = 'linear',C=10.0, random_state=1)
sv.fit(x_train,y_train)
SVC(C=10.0, kernel='linear', random_state=1)

y_pred = sv.predict(x_test)

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_true=y_test,y_pred=y_pred)
fig,ax = plt.subplots(figsize=(5,5))
ax.matshow(conf,cmap=plt.cm.Oranges,alpha=0.3)

for i in range(conf.shape[0]):
    for j in range(conf.shape[1]):
        ax.text(x-j,y=i,s=conf[i,j],va='center',size = 'xx-large')
plt.plot(y_test,y_pred)
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print("Precision: %.3f"%precision_score(y_test,y_pred))
print('Recall: %.3f'%recall_score(y_test,y_pred))
print('Accuracy: %3f'%accuracy_score(y_test,y_pred))
print('fl_score: %.3f'%f1_score(y_test,y_pred))
