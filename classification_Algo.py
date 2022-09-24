import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

d = pd.read_csv(r"C:\Users\sonas\Downloads\Social_Network_Ads (1)(1).csv")
x = d.iloc[:,:-1].values
y = d.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
mc = MinMaxScaler()
x_train = mc.fit_transform(x_train)
x_test = mc.transform(x_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5,metric='minkowski',p=2)
knn.fit(x_train,y_train)
print("Accuracy of KNN Classifier on training set: {:.2f}%".format(knn.score(x_train, y_train)*100))
print("Accuracy of KNN Classifier on testing set: {:.2f}%".format(knn.score(x_test, y_test)*100))
knn_score=knn.score(x_test,y_test)*100

#Naive bayes
from sklearn.naive_bayes import GaussianNB
gb= GaussianNB()
gb.fit(x_train,y_train)
print("Accuracy of Baye's Classifier on training set: {:.2f}%".format(gb.score(x_train, y_train)*100))
print("Accuracy of Baye's Classifier on testing set: {:.2f}%".format(gb.score(x_test, y_test)*100))
gb_score=gb.score(x_test,y_test)*100

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy",random_state=0)
dtc.fit(x_train,y_train)
print("Accuracy of DTree Classifier on training set: {:.2f}%".format(dtc.score(x_train, y_train)*100))
print("Accuracy of DTree Classifier on testing set: {:.2f}%".format(dtc.score(x_test, y_test)*100))
dtc_score=dtc.score(x_test,y_test)*100

#LogisticRegression
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train,y_train)
print("Accuracy of LReg Classifier on training set: {:.2f}%".format(lg.score(x_train, y_train)*100))
print("Accuracy of LReg Classifier on testing set: {:.2f}%".format(lg.score(x_test, y_test)*100))
lg_score = lg.score(x_test,y_test)*100

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)
rfc.fit(x_train,y_train)
print("Accuracy of RForest Classifier on training set: {:.2f}%".format(rfc.score(x_train, y_train)*100))
print("Accuracy of RForest Classifier on testing set: {:.2f}%".format(rfc.score(x_test, y_test)*100))
rfc_score = rfc.score(x_test,y_test)*100

#Support Vector Machine
from sklearn.svm import SVC
svc = SVC(kernel = 'linear',random_state=0)
svc.fit(x_train,y_train)
print("Accuracy of SVC Classifier on training set: {:.2f}%".format(svc.score(x_train, y_train)*100))
print("Accuracy of SVC Classifier on testing set: {:.2f}%".format(svc.score(x_test, y_test)*100))
svc_score = svc.score(x_test,y_test)*100

#GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=1000,max_depth = 3, min_samples_split=5, random_state=0)
gbr.fit(x_train,y_train)
print("Accuracy of GBR Classifier on training set: {:.2f}%".format(gbr.score(x_train, y_train)*100))
print("Accuracy of GBR Classifier on testing set: {:.2f}%".format(gbr.score(x_test, y_test)*100))
gbr_score = gbr.score(x_test,y_test)*100
gbr_score = round(gbr_score,2)

scores = [knn_score,gb_score, dtc_score,lg_score,rfc_score,svc_score,gbr_score]
algo = ['KNN', 'Naive bayes', 'decision tree', 'LogisticRegression', 'RandomForestClassifier', 'Support Vector Machine', 'GradientBoosting']
for i in range(len(algo)):
    print("The accuracy score achieved using {} is: {}%".format(algo[i],scores[i]))

import seaborn as sns
sns.set(rc={'figure.figsize':(10,7)})
plt.xlabel("algorithms")
plt.ylabel("Accuracy scores (in %)")
sns.barplot(algo,scores)
