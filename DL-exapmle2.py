#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:33:32 2021

@author: haizhou
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
X_trainval, X_test, y_trainval, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
print("Size of training set: {} size of validation set: {} size of test set: {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score = 0

#mlp = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
for n_hidden_nodes in [2, 5, 10, 50, 100]:
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1]:#alpha正则化 参数
        mlp=MLPClassifier(solver='lbfgs',random_state=0,
                          hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes],alpha=alpha)
        mlp.fit(X_trainval, y_trainval)
scores = cross_val_score(mlp, X_trainval, y_trainval, cv=5)
score = np.mean(scores)
if score > best_score:
    best_score = score
    best_parameters = {'alpha':alpha, 'hidden_layer_sizes': [n_hidden_nodes,n_hidden_nodes]}
   
# train model
mlp = MLPClassifier(solver='lbfgs', activation='relu', **best_parameters)
mlp.fit(X_trainval, y_trainval)

print("Train set score: {:.2f}".format(mlp.score(X_trainval, y_trainval)))

# evaluate the model and output the accuracy
print("Test set score: {:.2f}".format(mlp.score(X_test, y_test)))

