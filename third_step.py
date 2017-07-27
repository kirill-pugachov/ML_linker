# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:12:46 2017

@author: Kirill
"""

import sys
import pyprind
import csv
import re
#import scipy as sp
import numpy as np
#import pandas as pd
from sklearn.feature_extraction.text  import HashingVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.naive_bayes import GaussianNB
#import itertools
#from  sklearn.feature_extraction.text  import CountVectorizer


#Данные исходные для работы
file_path = 'C:/Users/Кирилл/Documents/MDM/Компендиум/Геоаптека/Links_data/'
file_name = 'LINKED.CSV'
file_name_1 = 'new_linked.txt'
file_name_2 = 'LINKED_2.CSV'
file_data = file_path + file_name
file_data_1 = file_path + file_name_1
file_data_2 = file_path + file_name_2



def get_all_classes(file_data):
    result = set()
    with open(file_data) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            result.add(int(line[2]))
#            if len(list(result)) == 45000:
#                return list(result)
#                break
    return list(result)              

    
def tokenizer(text):
    text = re.sub('[\W]+', ' ', text.lower())
    tokenized = [w for w in text.split()]    
    return tokenized


def stream_docs(path):
    with open(path) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            text, label = line[1], int(line[2])
            yield text, label    
           
            
def get_minibatch(doc_stream, size):
    docs, y, mark = [], [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
            mark.append(label)
    except StopIteration:
        return None, None, None
    return docs, y, mark

    
def stream_docs_1(path):
    with open(path) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            text, label = line[1], int(line[2])
            yield text, label 

    
def get_minibatch_1(doc_stream, size):
    docs, y, mark = [], [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
            mark.append(label)
    except StopIteration:
        return None, None, None
    return docs, y, mark
    
    
    
vect = HashingVectorizer(decode_error = 'ignore', n_features = 2**21, preprocessor = None, tokenizer = tokenizer)
clf = SGDClassifier(loss = 'log', penalty='l2', n_iter = 3)
doc_stream = stream_docs(file_data)
#classes = get_all_classes(file_data_2)
print('Посчитали классы')


X_train, y_train, classes = get_minibatch(doc_stream, size = 15)
X_train_proc = vect.transform(X_train)
#clf.fit(X_train_proc, y_train)
clf.partial_fit(X_train_proc, y_train, classes = classes)
parts = 15

while parts <= 1000000:
    X_train, y_train, classes = get_minibatch(doc_stream, size = 15)
    if X_train:
        X_train_proc = vect.transform(X_train)
        clf.partial_fit(X_train_proc, y_train)
        parts += 15
    else:
        break
print(parts)    

doc_stream_1 = stream_docs_1(file_data_2)
X_test, y_test, classes = get_minibatch_1(doc_stream_1, size = 20)
X_test = vect.transform(X_test)   
print('Верность: %.3f' % clf.score(X_test, y_test)) 