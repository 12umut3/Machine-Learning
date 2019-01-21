#===Import Necessary libraries========#

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


X=X4
y=y4

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
print 'accuracy of model:',accuracy

pred= np.array(pred)
y= np.array(y)



confusion_matrix= pd.crosstab(y,pred,rownames=['Actual'], colnames=['Predicted'], margins=True)

print 'Confusion Matrix:',confusion_matrix

TP= confusion_matrix.iloc[0:1,0:1].values
FN= confusion_matrix.iloc[1:2,0:1].values
FP= confusion_matrix.iloc[0:1,1:2].values
Precision=np.float(np.float(TP)/(TP+FP))
Recall=np.float(np.float(TP)/(TP+FN))
print 'Precision:',Precision
print 'Recall :',Recall

import matplotlib.pyplot as plt
plt.plot([0.5,0.2,0.5,0.5,0.5],[0.5,1,0.5,0.5,0.5],'r--')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()












