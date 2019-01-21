import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
data=pd.read_csv('UCI_Breast_Cancer.csv')
Y=data.iloc[:,10:11].values
X=data.iloc[:,0:10].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.333,random_state=42)
from sklearn import svm
clf=svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
clf.fit(X,Y)
pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(pred,y_test)
plt.scatter(pred,y_test)
plt.plot(pred,y_test,'--r')
plt.show()

