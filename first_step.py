# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
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
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


vect = HashingVectorizer(decode_error = 'ignore', n_features = 2**21, preprocessor = None, tokenizer = tokenizer)
clf = SGDClassifier(loss = 'log', penalty='l2', n_iter = 1)
#clf = GaussianNB()
doc_stream = stream_docs(file_data_2)
pbar = pyprind.ProgBar(50)
#classes = np.array(get_all_classes(file_data_2))
classes = get_all_classes(file_data_2)
print('Посчитали классы')

X_train, y_train = get_minibatch(doc_stream, size = 10)
X_train = vect.transform(X_train)
clf.partial_fit(X_train.todense(), y_train, classes = classes)
print('Первая загрузка данных в модель')

for _ in range(50):
    X_train, y_train = get_minibatch(doc_stream, size = 10)
    if not X_train:
        break
    X_train = vect.transform(X_train)
#    X_train_1 = X_train.toarray()
#    y_train_1 = np.array(y_train)
#    print(np.unique(y_train_1))
    clf.partial_fit(X_train.todense(), y_train)   
    pbar.update()
X_test, y_test = get_minibatch(doc_stream, size = 50)
X_test = vect.transform(X_test)   
print('Верность: %.3f' % clf.score(X_test, y_test)) 
    
#            
#with open(file_data) as file_key:
#    reader = csv.reader(file_key, delimiter=';')
#    for line in reader:
#        print(tokenizer(line[1]), line[3], '\n')
##    drug_id_names = get_dict_id(reader)
##    drug_id = create_drug_id_dict(reader)   
##print(len(drug_id[0]), len(drug_id[1]), 'Все посчитали')
####print('Все посчитали')
##chancker = pd.read_csv(file_data, sep = ';', skiprows = 0, chunksize=1000)
###chunker = pd.read_table(file_data, chunksize=1000)
##for chunk in chancker:
##    print(chunk.info())
#
##data = sp.genfromtxt(file_data, skip_header=1, usecols=(1, 2), delimiter = ';', max_rows=1000)
#
##x_data = data[:,0]
##y_data = data[:,1]
##print('считали данные из csv файла')
##print(len(x_data), len(y_data))
##vectorizer = CountVectorizer(min_df=l) 
#
#
##def get_dict_id(reader):
##    result = dict()
##    for line in reader:
###        print(line)
##        if line[2] not in result:
###            print(line[2], line[3])
##            result[line[2]] = line[3]  
##    return result
#
#def get_all_classes_1(file_data):
#    result = set()
#    with open(file_data) as file_key:
#        reader = csv.reader(file_key, delimiter=';')
#        next(reader)
#        for line in reader:
#            result.add(int(line[1]))
#            if len(list(result)) == 45000:
#                return list(result)
#                break
#    return list(result)
#                
#    
#def stream_docs_1(path):
#    with open(path) as file_key:
#        reader = csv.reader(file_key, delimiter=';')
#        next(reader)
#        for line in reader:
#            text, label = line[0], int(line[1])
#            yield text, label            
#def create_drug_id_dict(reader):
#    '''Разведка что и где и в каком кол-ве
#    где данные, а где метки и сколько всего 
#    строк'''
#    result = dict()
#    result_1 = dict()
#    for line in reader:
#        if line[2] in result:
#            result[line[2]] += 1
#        else:
#            result[line[2]] = 1
#            result_1[line[2]] = line[3]
#    return (result, result_1)   
#def split_file(file_path, file_name):
#    number = 1
#    file_data = file_path + file_name
#    with open(file_data) as file_key:
#        reader = csv.reader(file_key, delimiter=';')
#        next(reader)
#        for line in reader:
#            number +=1
#            if number % 100 == 0:
#                original = sys.stdout
#                sys.stdout = open(file_path + 'new_linked.csv', 'a')
#                print('"' + line[1] + '"' + ';' + line[2] + '\n' + '\r')
#                sys.stdout = original         