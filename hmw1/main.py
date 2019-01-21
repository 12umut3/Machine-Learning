import csv
from array import *
import math
import numpy as np

#########TRAIN#############
train_features = list(csv.reader(open('question-4-train-features.csv')))
train_labels = list(csv.reader(open('question-4-train-labels.csv')))


numrows = len(train_features)    
numcols = len(train_features[0])
#print(numrows)
#print(numcols)

spaceEstimate = [[0 for i in range(1)]  for j in range(numcols)]
medicalEstimate = [[0 for i in range(1)]  for j in range(numcols)]

#estimates
totalSpaceWordNumber = 0;
totalMedicalWordNumber = 0;


#calculate estimations
#question 4.5 for MLE
for j in range(0,numcols-1):
	for i in range(0,numrows-1):
		if train_labels[i][0] == '0': #for each madical newsmail
			medicalEstimate[j][0] = medicalEstimate[j][0] + int(train_features[i][j]) 
			totalMedicalWordNumber = totalMedicalWordNumber + int(train_features[i][j]) + 1
		elif train_labels[i][0] == '1': # for each space document
			spaceEstimate[j][0] = spaceEstimate[j][0] + int(train_features[i][j]) 
			totalSpaceWordNumber =  (totalSpaceWordNumber + int(train_features[i][j])) + 1

#calculation of MLE
medicalEstimate  = np.divide(medicalEstimate,  totalMedicalWordNumber+(numcols*numrows))
spaceEstimate  = np.divide(spaceEstimate,  totalSpaceWordNumber + (numrows*numcols))
########TEST##############
test_features = list(csv.reader(open('question-4-test-features.csv')))
test_labels = list(csv.reader(open('question-4-test-labels.csv')))

numrows1 = len(test_labels)    
numcols1 = len(test_labels[0])

medical_number = 0;
space_number = 0;
medical_probability = 0.0
space_probability = 0.0

for i in range (0,numrows1-1):
	if test_labels[i][0] == '0':
		medical_number = medical_number + 1
	elif test_labels[i][0] == '1':
		space_number = space_number + 1

x = [[0 for i in range(1)]  for j in range(numrows1)]
y = [[0 for i in range(2)]  for j in range(2)]

for i in range (0,numrows1-1):
	space_probability = math.log10(space_number/(space_number + medical_number)) 
	medical_probability = math.log10(medical_number/(space_number + medical_number))
	for j in range(0,numcols1):
		if test_features[i][j] != '0':
			if medicalEstimate[j][0] != 0:
				medical_probability = medical_probability + int(test_features[i][j]) * math.log(medicalEstimate[j][0])
				#print(medical_probability)
				space_probability = space_probability + int(test_features[i][j]) * math.log(spaceEstimate[j][0])
				#print(space_probability)
	if medical_probability >= space_probability:
		x[i][0] = 0
	elif medical_probability < space_probability:
		x[i][0] = 1

        

#Calculate accuracy
for i in range (0,numrows1-1):
	if x[i][0] == 1 and test_labels[i][0] == '1':
		y[0][0] = y[0][0] +1;
	elif x[i][0] == 0 and test_labels[i][0] == '0':
		y[1][1] = y[1][1] +1;
	elif x[i][0] == 1 and test_labels[i][0] == '0': 
		y[0][1] = y[0][1] +1;
	elif x[i][0] == 0 and test_labels[i][0] == '1':
		y[1][0] = y[1][0] +1;
	
		
accurate = y[0][0] + y[1][1]
accuracy = accurate / 400;

print(accuracy)



































