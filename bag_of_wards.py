# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:22:13 2017

@author: Kirill
"""

#bag_of_wards aproach
import shelve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Данные исходные для работы
file_path = 'C:/Users/Кирилл/Documents/MDM/Компендиум/Геоаптека/Links_data/LINKED/'
file_name = 'LINKED.CSV'
file_data = file_path + file_name
name_shelve_data = 'data_base'
name_shelve_data_1 = 'models_base_db'

if __name__ == '__main__':
    data_values = []
    db = shelve.open(name_shelve_data, writeback=True)
    count = CountVectorizer()
    tfidf = TfidfTransformer
    for key in db.keys():
        data_values.extend(db[key])
    bag = count.fit_transform(data_values)
#    tfidf.fit_transform(bag)
    
    print(count.vocabulary_)
