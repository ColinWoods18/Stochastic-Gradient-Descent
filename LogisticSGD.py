import pandas as pd
import numpy as np
import random as rd
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt

#for commit to git

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

y = y.to_numpy()

test_y = test_norm['HasDiabetes']

test_y = test_y.to_numpy()

train_norm = train_norm.to_numpy()
test_norm = test_norm.to_numpy()

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + np.exp(-yhat))
 
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, alpha, iterations):
    coef = [0.0 for i in range(len(train[0]))]
    l2NormArray = list()
    SSEarray = list()
    lossArray = list()
    for epoch in range(iterations):     

        for row in train:
            yhat = predict(row, coef)           
            error = row[-1] - yhat
            coef[0] = coef[0] + alpha * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + alpha * error * yhat * (1.0 - yhat) * row[i]

        lossArr = yhat - test_y   
        if(np.mod(epoch, 100) == 0):
            l2NormArray.append(l2Norm(coef))   
            SSE = np.sum((yhat - test_y)**2) 
            SSEarray.append(SSE)  
            loss = np.sum((lossArr**2)/iterations)
            lossArray.append(loss)

    return coef #"Loss: ", lossArray, "\n\n l2Norm: ", l2NormArray, "\n\n SSE: ", SSEarray
 
def l2Norm(coef):
    l2 = np.sqrt(np.sum(coef)**2)
    return l2


def logistic_SGD(train, test, alpha, iterations):
    predictions = list()
    coef = coefficients_sgd(train, alpha, iterations)
    for row in test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return(predictions)


print("Logistic Regression Coeffs:")
print(coefficients_sgd(train_norm, 0.00001, 10000))


