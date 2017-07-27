# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 07:42:26 2017

@author: Kirill
"""
import csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


#Данные исходные для работы
file_path = 'C:/Users/Кирилл/Documents/MDM/Компендиум/Геоаптека/Links_data/'
file_name = 'LINKED.CSV'
file_data = file_path + file_name

def stream_docs(path):
    with open(path) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            text, label = line[1], int(line[2])
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


def stream_docs_1(path):
    with open(path) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            text, label = line[1], int(line[2])
            yield text, label     
    
def get_minibatch_by_step(doc_stream, size, step):
    docs, y = [], []
    try:
        for _ in range(0, size, step):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

    
def prepare_y(y_temp):
    y_result = []
    for element in y_temp:
        y_result.append([str(element)])       
    return y_result
    

if __name__ == '__main__':
    doc_stream = stream_docs(file_data)
    X_temp, y_temp = get_minibatch(doc_stream, size = 6000)
    

    X_train = np.array(X_temp)
    y_train_text = prepare_y(y_temp)
    
#    doc_stream_1 = stream_docs_1(file_data)
    X_test, y_test_temp = get_minibatch_by_step(doc_stream, size = 9000, step = 3)


    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_train_text)
    
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])
    
    classifier.fit(X_train, Y)
    predicted = classifier.predict(X_test)
    all_labels = mlb.inverse_transform(predicted)
    
    for item, labels, y in zip(X_test, all_labels, y_test_temp):
        print('{0} => {1} => {2}'.format(item, ', '.join(labels), y))