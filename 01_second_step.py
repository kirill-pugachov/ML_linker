# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:57:42 2017

@author: Kirill
"""

#import sys
#from collections import Counter
#import pyprind
import csv
import re
#import scipy as sp
import numpy as np
#import pandas as pd
from sklearn.feature_extraction.text  import HashingVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.naive_bayes import GaussianNB
#import itertools
#from  sklearn.feature_extraction.text import CountVectorizer


#Данные исходные для работы
file_path = 'C:/Users/Кирилл/Documents/MDM/Компендиум/Геоаптека/Links_data/LINKED/'
file_name = 'LINKED.CSV'
file_data = file_path + file_name


def get_all_classes(file_data):
    result = set()
    result_1 = set()
    with open(file_data) as file_key:
#        length = Counter(file_key)
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            if line[2].split('.')[2] == '2017':
#                print(line[2].split('.')[2])
                result.add(int(line[3]))
#                print(len(result))

#    print(len(result))
    return list(result)            


def create_drug_id_dict(reader):
    '''Разведка что и где и в каком кол-ве
    где данные, а где метки и сколько всего 
    строк'''
    result = dict()
    result_1 = dict()
    for line in reader:
        if line[2] in result:
            result[line[2]] += 1
        else:
            result[line[2]] = 1
            result_1[line[2]] = line[3]
    return (result, result_1)    
    
    
def file_input_size(file_data):
    length = 0
    with open(file_data) as file_key:
        for raw in file_key:
            length += 1
    return length
        
    
def tokenizer(text):
    text = re.sub('[\W]+', ' ', text.lower())
    tokenized = [w for w in text.split()]    
    return tokenized


def stream_docs(path):
    with open(path) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            text, label = line[1], str(line[3])
            yield text, label    
            
            
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

    
if __name__ == '__main__':
    result = get_all_classes(file_data)
    

    vect = HashingVectorizer(decode_error = 'ignore', n_features = 2**17, preprocessor = None, tokenizer = tokenizer)
    clf = SGDClassifier(loss = 'log', penalty='l2', n_iter = 1)
    #clf = GaussianNB()
    doc_stream = stream_docs(file_data)
    #pbar = pyprind.ProgBar(95)
    classes = np.array(get_all_classes(file_data))
    X_train, y_train = get_minibatch(doc_stream, size = 10)
    X_train = vect.transform(X_train)
    #    X_train_1 = X_train.toarray()
    #    y_train_1 = np.array(y_train)
    #    print(np.unique(y_train_1))
    clf.partial_fit(X_train, y_train, classes = classes)    
    
    
    for _ in range(94):
        X_train, y_train = get_minibatch(doc_stream, size = 10)
        if not X_train:
            break
        X_train = vect.transform(X_train)
    #    X_train_1 = X_train.toarray()
    #    y_train_1 = np.array(y_train)
    #    print(np.unique(y_train_1))
        clf.partial_fit(X_train, y_train)   #X_train.todense()
        X_train.flush()
        y_train.flush()
        classes.flush()
        
        
        
    #    pbar.update()
    X_test, y_test = get_minibatch(doc_stream, size = 50)
    X_test = vect.transform(X_test)   
    print('Верность: %.3f' % clf.score(X_test, y_test)) 