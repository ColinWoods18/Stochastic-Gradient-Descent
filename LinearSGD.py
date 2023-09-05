import pandas as pd
import numpy as np
import random as rd
from random import seed
from random import randrange
from csv import reader
from math import sqrt


data = pd.read_csv("pima-indians-diabetes.csv")
learning_rate = 0

train_sample = data.sample(500, random_state = 25)
test_sample = data.drop(train_sample.index)

train_norm = train_sample.copy()
test_norm = test_sample.copy()
for feature in train_norm.columns:
    max = train_norm[feature].max()
    min = train_norm[feature].min()
    train_norm[feature] = (train_norm[feature] - min) / (max - min)
    test_norm[feature] = (test_norm[feature] - min)/ (max - min)



y = train_norm['HasDiabetes']
x = train_norm.drop('HasDiabetes', axis= 1)

test_y = test_norm['HasDiabetes']

test_y = test_y.to_numpy()


train_norm = train_norm.to_numpy()
test_norm = test_norm.to_numpy()


def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat
 

def coefficientsSGD(train, alpha, iterations):
	coef = [0.0 for i in range(len(train[0]))]
	l2NormArray = list()
	SSEarray = list()
	lossArray = list()

	for epoch in range(iterations):
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			coef[0] = coef[0] - alpha * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - alpha * error * row[i]
		lossArr = yhat - test_y   
		if(np.mod(epoch, 100) == 0):
			l2NormArray.append(l2Norm(coef))   
			SSE = np.sum((yhat - test_y)**2) 
			SSEarray.append(SSE)  
			loss = np.sum((lossArr**2)/iterations)
			lossArray.append(loss)
	return "Loss: ", lossArray, "L2Norm: ", l2NormArray, "SSE: ", SSEarray

def l2Norm(coef):
    l2 = np.sqrt(np.sum(coef)**2)
    return l2
 
# LinearSGD
def linearSGD(train, test, alpha, iterations):
	predictions = list()
	coef = coefficientsSGD(train, alpha, iterations)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return(predictions)

print(coefficientsSGD(train_norm, 0.00001, 10000))

#print(linearSGD(train_norm, test_norm, 0.8, 1))


 



