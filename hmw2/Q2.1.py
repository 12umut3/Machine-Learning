#===Import Necessary libraries========#

import pandas as pd
import numpy as np
import time

def sigmoid(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient

def log_likelihood(x, y, weights):
    z = np.dot(x, weights)
    ll = np.sum( y*z - np.log(1 + np.exp(z)) )
    return ll

def gradient_ascent(X, h, y):
    return np.dot(X.T, y - h)
def update_weight_mle(weight, learning_rate, gradient):
    return weight + learning_rate * gradient

X=pd.read_csv('ovariancancer.csv')
y=pd.read_csv('ovariancancer_labels.csv')
#y=y.iloc[:,0]

#==Five Fold===#
X1= X.iloc[0:43,1:]
X2= X.iloc[43:86,1:]
X3= X.iloc[86:129,1:]
X4=X.iloc[129:172,1:]
X5= X.iloc[172:215,1:]
y1=y.iloc[0:43,0]
y2=y.iloc[43:86,0]
y3=y.iloc[86:129,0]
y4=y.iloc[129:172,0]
y5=y.iloc[172:215,0]

XX=[X1,X2,X3,X4,X5]
yy=[y1,y2,y3,y4,y5]

CV_fold_Accuracy=[]
for i in range(0,5):
	X=XX[i]
	y=yy[i]



	intercept = np.ones((X.shape[0], 1)) 
	X = np.concatenate((intercept, X), axis=1)
	theta = np.zeros(X.shape[1])

	num_iter=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
	learning_rate=[0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
	for i in num_iter:
		for j in learning_rate:
			h= sigmoid(X, theta)
			gradient = gradient_descent(X, h, y)
			theta = update_weight_loss(theta, j, gradient)

	result = sigmoid(X, theta)
	f = pd.DataFrame(np.around(result, decimals=6)).join(y)

	pred= f[0].apply(lambda x : 0 if x < 0.5 else 1)
	acc=[1 for i in range(0,y.shape[0]) if pred[i]==y.iloc[i]]
	accuracy= (np.float(len(acc))/(y.shape[0]))*100
	CV_fold_Accuracy.append(accuracy)
	print 'Mean accuracy of model:',accuracy



print "Accuracy of each fold :",CV_fold_Accuracy
print 'Mean Accuracy :',np.mean(CV_fold_Accuracy)







