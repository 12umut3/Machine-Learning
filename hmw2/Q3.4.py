import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
data=pd.read_csv('UCI_Breast_Cancer.csv')
Y=data.iloc[:,10:11].values
X=data.iloc[:,0:10].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.333,random_state=42)

accuracy_c=[]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] 
for i in gamma:

	from sklearn.svm import SVC
	clf=SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
	    	decision_function_shape='ovr', degree=3, gamma=i, kernel='rbf',
	    	max_iter=-1, probability=False, random_state=None, shrinking=True,
	    	tol=0.001, verbose=False)
	clf.fit(X_train,y_train)
	pred=clf.predict(X_test)
	from sklearn.metrics import accuracy_score
	accuracy=accuracy_score(pred,y_test)
	accuracy_c.append(accuracy)
	print accuracy

plt.plot(gamma,accuracy_c)
plt.xlabel('gamma')
plt.ylabel('accuracy')
plt.show()

