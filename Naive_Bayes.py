# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:47:07 2017

@author: Kirill
"""

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
y_pred_mnb = mnb.fit(iris.data, iris.target).predict(iris.data)
y_pred_bnb = bnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))
print("Number of mislabeled points MNB out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred_mnb).sum()))
print("Number of mislabeled points BNB out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred_bnb).sum()))