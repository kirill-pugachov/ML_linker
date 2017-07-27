# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:36:45 2017

@author: Kirill
"""

import numpy as np
X = np.random.randint(6, size=(7, 100))
y = np.array([1, 2, 3, 4, 5, 6, 7])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))