# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:09:35 2017

@author: Kirill
"""

#цсв файл на входе
#разбираем по присвоенным drug_id в словарь - ключ - список строк
#складываем в шелв_1
#из шелва достаем по ключу и тренируем по отдельному ключу отдельную модель
#все модели складываем в шелв по ключу
#берем списко строк из тестового набора и прогоняем через распознавание по всему шелву
#там где нуль - игнорим, там где больше нуля пишем в словарь результатов drug_id - значение распознования - 
#строка тестовая - строка известная
#

import csv
import shelve
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
name_shelve_data = 'data_base'
name_shelve_data_1 = 'models_base_db'

def dict_docs(path):
    result_dict = dict()
    with open(path) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            if line[2] in result_dict.keys():
                result_dict[line[2]] += 1#.append(line[1])
            else:
                result_dict[line[2]] = 1#[line[1]]
    return result_dict


def more_then_one(dict_docs):
    result = 0
    for key in dict_docs.keys():
        if dict_docs[key] > 1:
            result += 1
    return result


def create_data_sets(path):
    with shelve.open(name_shelve_data, writeback=True) as db:
        with open(path) as file_key:
            reader = csv.reader(file_key, delimiter=';')
            next(reader)
            for line in reader:
                if line[2] in db:
                    db[line[2]].append(line[1])
                    db.sync()
                else:
                    db[line[2]] = [line[1]]
                    db.sync()
    db.close()
                    

def prepare_y(y_temp):
    y_result = []
    for element in y_temp:
        y_result.append([str(element)])       
    return y_result    
    
def create_data_set(raws_list, dict_key):
    X_temp_1 = raws_list
    y_temp_1 = list()
    for raws in raws_list:
        y_temp_1.append(dict_key)
    return X_temp_1, y_temp_1
            
if __name__ == '__main__':
#    res_sorted = dict_docs(file_data)
#    print(len(res_sorted))
#    print(max(res_sorted.values()))
#    print(more_then_one(res_sorted))
##    create_data_sets(file_data)
#Обучаем модели    
    mlb = MultiLabelBinarizer()
    classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(LinearSVC()))])
    
    with shelve.open(name_shelve_data, writeback=True) as db:    
        for key in db.keys():
            if len(db[key]) > 2:
                X_temp, y_temp = create_data_set(db[key], key)
                X_train = np.array(X_temp)
                y_train_text = prepare_y(y_temp)
                Y = mlb.fit_transform(y_train_text)
#                classifier.fit(X_train, Y)
                with shelve.open(name_shelve_data_1, writeback=True) as model_db:
                    if key in model_db:
                        print('ДВА ОДИНАКОВЫХ DRUG_ID')
                        break
                    else:
                        model_db[key] = classifier.fit(X_train, Y)
                        model_db.sync()
                    model_db.close()
        db.sync()
        db.close()