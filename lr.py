import numpy as np
import random
import pandas as pd 
import h5py

def Loss(y,yhat):
	error = 0
	for i in range(len(y)):	
		if y[i] == 0:
			error += -(np.log(1 - yhat[0][i]))
		elif y[i] == 1:
			error += -(np.log(yhat[0][i]))
	return error/len(y)

def sigmoid(Z):
	return (1/(1+np.exp(-Z)))

train_dataset = h5py.File('train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

test_dataset = h5py.File('test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

# train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
# test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255

X = train_set_x
Y = train_set_y_orig

n_x = X.shape[0]
m = X.shape[1]
epochs = 100
# X = X.T
# W = random.randn((n_x, 1))
W = np.zeros((n_x,1))
b = 0
J = 0
# dW = np.zeros((n_x,1))
alpha = 0.0001

for epoch in range(epochs):
	Z = np.dot(W.T, X) + b
	A = sigmoid(Z)
	dZ = A-Y
	dW = (np.dot(X, dZ.T))
	db = (np.sum(dZ))

	W -= alpha*dW
	b -= alpha*db
	print(Loss(Y,A))



