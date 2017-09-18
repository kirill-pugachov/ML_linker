# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:22:13 2017

@author: Kirill
"""

#bag_of_wards aproach
#import time
#import sys
import csv
import shelve
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


#Данные исходные для работы
file_path = 'C:/Users/Кирилл/Documents/MDM/Компендиум/Геоаптека/Links_data/LINKED/'
file_name = 'LINKED.CSV'
file_data = file_path + file_name
name_shelve_data = 'data_base'
size_max = 11
direction_db_names = ['1', '2', '3']
#name_shelve_data_1 = 'models_base_db'

def build_data_2017(file_data):
    '''All data from 2017 drug_id: [line..., ...]'''
    count_line = 0
    count_line_1 = 0
    with shelve.open(name_shelve_data, writeback=True) as db:    
        with open(file_data) as file_key:
            reader = csv.reader(file_key, delimiter=';')
            next(reader)
            for line in reader:
                if line[2].split('.')[2] == '2017':
                    count_line_1 += 1
                    if line[3] in db:
                        db[line[3]].append(line[1])
                    else:
                        db[line[3]] = [line[1]]
                count_line += 1
            db.sync()
    return count_line, count_line_1

    
def build_data_2017_1(file_data):
    '''All data from 2017 drug_id: [line..., ...]'''
    result_dict = dict()
    count_line = 0
    count_line_1 = 0
    with shelve.open(name_shelve_data, writeback=True) as db:    
        with open(file_data) as file_key:
            reader = csv.reader(file_key, delimiter=';')
            next(reader)
            for line in reader:
                if line[2].split('.')[2] == '2017':
                    count_line_1 += 1
                    if line[3] in result_dict:
                        result_dict[line[3]].append(line[1])
                    else:
                        if line[3]:
                            result_dict[line[3]] = [line[1]]
                        else:
                            print('Строка без данных: ', line, '\n')
                count_line += 1
        db['0'] = result_dict
        db.sync()
    return count_line, count_line_1    
    

def get_right_name(file_data, direction_vector):
    '''All right names of data from 2017 drug_id and direction_vector'''
    right_name_dict = dict()
    with open(file_data) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
                if line[2].split('.')[2] == '2017':
                    if line[3] in direction_vector:
                        if line[3] in right_name_dict:
                            continue 
                        else:
                            if line[3]:
                                right_name_dict[line[3]] = line[4]
                            else:
                                print('Строка без данных: ', line, '\n')
    return right_name_dict


def read_list(y_test):
    '''read the list'''
    for line in y_test:
        label = line
        yield label
        
        
def get_minibatch(doc_stream, size):
    '''read the list by batches'''
    y = []
    try:
        for _ in range(size):
            label = next(doc_stream)
            y.append(label)
    except StopIteration:
        return None
    return y

def stream_docs_test(file_data, direction_vector):
    with open(file_data) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            if line[2].split('.')[2] == '2016':
                if line[3] in direction_vector:
                    yield line[1], line[3]    
       

def test_data_build(file_data, direction_vector):
    '''build test data by 2016 mark and direction_vector to separate train and test data'''
    result_dict = dict()
    with open(file_data) as file_key:
        reader = csv.reader(file_key, delimiter=';')
        next(reader)
        for line in reader:
            if line[2].split('.')[2] == '2016':
                if line[3] in direction_vector:
                    if line[3] in result_dict:
                        result_dict[line[3]].append(line[1])
                    else:
                        if line[3]:
                            result_dict[line[3]] = [line[1]]
                        else:
                            print('Строка без данных: ', line, '\n')
    return result_dict
                            

def build_model(direction_vector, size_max, get_minibatch, data_read):
    result = list()
    result_cls = list()
    
    for _ in range(len(direction_vector)//size_max):
        vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
        classifier = SGDClassifier(loss='log', warm_start=True, n_jobs=-1)    
        with shelve.open(name_shelve_data) as db:
            drug_id_list = get_minibatch(data_read, size_max)
            for drug in drug_id_list:        
                X_train = vectorizer.transform(db['0'][drug])
                y_train = np.array([int(drug)]*len(db['0'][drug]))        
                classifier.partial_fit(X_train, y_train, classes = [int(drug) for drug in drug_id_list])
        result_cls.append(classifier)
        _ = joblib.dump(classifier, str(_), compress=9)
#        _ = joblib.dump(vectorizer, drug + '_' + 'vectorizer', compress=9)
        result.append(drug)
        del vectorizer
        del classifier
        del X_train
        del y_train
    return result, result_cls


def get_direction_vector(name_shelve_data):
    '''get all classes marks from dict in shelve'''
    with shelve.open(name_shelve_data, writeback=True) as db:
        direction_vector = db['0'].keys()
    return direction_vector
    

def get_test_data(test_data_dict):
    '''build test data'''
    X_test = []
    y_test = []
    y_test_2 = []
    for key in test_data_dict:
        y_test_2.append(int(key))
        y_test += ([int(key)] * len(test_data_dict[key]))
        X_test += test_data_dict[key]
    return X_test, y_test, y_test_2


def get_classifier(cls):
    return joblib.load(cls)


def save_directions(direction_vector, direction_vector_names, model_names, direction_db_names):
    with shelve.open(name_shelve_data, writeback=True) as db:
        db[direction_db_names[0]] = direction_vector
        db[direction_db_names[1]] = direction_vector_names
        db[direction_db_names[2]] = model_names


def classifier_list_build(model_names):
    result = list()
    for name in model_names:
        result.append(get_classifier(name))
    return result
    
    
def check_quality(cls_list):
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
    with open('data_visual_compare', 'a') as file:
        writer = csv.writer(file, delimiter=";")
        data_from_file = stream_docs_test(file_data, direction_vector)
        temp = next(data_from_file)
        while temp:
            score_result = []
            score_result_0 = dict()
            x = temp[0]
            y = temp[1]
            
            for classifier in cls_list:
                score_result.append(int(bool(classifier.predict(vectorizer.transform([x])).tolist()[0] == y)))
                score_result_0[max(classifier.predict_proba(vectorizer.transform([x]))[0])] = classifier.predict(vectorizer.transform([x])).tolist()[0]
            writer.writerow([x, max(score_result_0.keys()), score_result_0[max(score_result_0.keys())], y, direction_vector_names[str(y)]])
            try:
                temp = next(data_from_file)
            except StopIteration:
                return score_result_0
    
if __name__ == '__main__':
    
    direction_vector = list(get_direction_vector(name_shelve_data))[0:49*size_max]
    data_read = read_list(direction_vector)
    model_names, cls_list = build_model(direction_vector, size_max, get_minibatch, data_read)       
    direction_vector_names = get_right_name(file_data, direction_vector)
    save_directions(direction_vector, direction_vector_names, model_names, direction_db_names)
#    model_names_1 = classifier_list_build(model_names)
    score_result_0 = check_quality(cls_list)





#######################################################
#    print("Оценка №1 Правильность на тестовом наборе: {:.2f}".format(sum(score_result)/len(score_result)))
#    print("Оценка №2 Правильность на тестовом наборе: {:.2f}".format(sum(score_result_1)/len(score_result_1)))

#    get_minibatch(data_read, size_max)
    
#    direction_vector_int = [int(drug) for drug in direction_vector]
#    for _ in range(len(direction_vector)//size_max):
#
#        vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
#        classifier = SGDClassifier(loss='log', warm_start=True, n_jobs=-1)
#    
#        with shelve.open(name_shelve_data) as db:
#            drug_id_list = get_minibatch(data_read, size_max)
#            for drug in drug_id_list:
#        
#                X_train = vectorizer.transform(db['0'][drug])
#                y_train = np.array([int(drug)]*len(db['0'][drug]))
#        
#                classifier.partial_fit(X_train, y_train, classes = [int(drug) for drug in drug_id_list])
#         
#            
#        _ = joblib.dump(classifier, drug, compress=9)
#        _ = joblib.dump(vectorizer, drug + '_' + 'vectorizer', compress=9)
##        with shelve.open(name_shelve_data) as db:
##            db[drug] = (vectorizer, classifier)
##            db.sync()
#        
#        del vectorizer
#        del classifier
#        del X_train
#        del y_train
#        print('Из аптеки: {0} Drug_id распознанное: {1} Drug_id из данных: {2} Написание из базы: {3}'.format(x, classifier.predict(vectorizer.transform([x])).tolist(), y, direction_vector_names[str(y)]), '\n')
#    sys.stdout = original   
#    original = sys.stdout
#    sys.stdout = open('data_visual_compare', 'a')
#    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
#    with open('data_visual_compare', 'a') as file:
#        writer = csv.writer(file, delimiter=";")
#        kkk = stream_docs_test(file_data, direction_vector)
#        temp = next(kkk)
#        while temp:
#            score_result = []
#            score_result_1 = []
#            x = temp[0]
#            y = temp[1]
#            for cls in model_names:
#                classifier = get_classifier(cls)
#                score_result.append(classifier.score(vectorizer.transform([x]), [y]))
#                score_result_1.append(accuracy_score(classifier.predict(vectorizer.transform([x])).tolist(), [y]))
#            writer.writerow([x, classifier.predict(vectorizer.transform([x])).tolist()[0], y, direction_vector_names[str(y)]])    
#            temp = next(kkk)
#    
#    X_test, y_test, y_test_1 = get_test_data(test_data_build(file_data, direction_vector))
#    score_result = []
#    score_result_1 = []

#    with open('data_visual_compare', 'a') as file:
#        writer = csv.writer(file, delimiter=";")
#        for x, y in zip(X_test, y_test):
#            for cls in model_names:
#                classifier = get_classifier(cls)
#                
#               
#                score_result.append(classifier.score(vectorizer.transform([x]), [y]))
#                score_result_1.append(accuracy_score(classifier.predict(vectorizer.transform([x])).tolist(), [y]))
#                writer.writerow([x, classifier.predict(vectorizer.transform([x])).tolist()[0], y, direction_vector_names[str(y)]])
#    direction_vector = ['423411', '304166', '165534', '80185', '13511', '121151', '101905', '108966', '92903', '107116', '115638', '392517', '308892', '124974', '68689', '83367', '38378', '133753', '301151', '98255', '438649', '54287', '12039', '73890', '86770', '167712', '148483', '296506', '84905', '333260', '45470', '59819', '18409', '9786', '49173', '77712', '437501', '86630', '352898', '119052', '376024', '442893', '452947', '138555', '110940', '200334', '102207', '352933', '353035', '68786', '232668', '296918', '96949', '300528', '90398', '388856', '50166', '394968', '418165', '82052',  '240581', '277637', '98309', '302523', '99964', '28616', '109694', '113265', '144242', '143510', '367091', '407806', '135034', '118311', '434508', '156473', '32772', '114033', '399890', '370656', '427293', '438786', '11376', '390638',  '175740', '109691', '137247', '165760', '88687', '77787', '235121', '232578', '47273', '426723', '146374', '199481', '5709', '85126', '19776', '110004', '272599', '197926', '196630', '79964', '79478', '72835', '136632', '283878', '87764', '461014', '77170', '436708', '124427', '302737', '138522', '130892', '430400', '285895', '179139', '247988', '113481', '150935', '329799', '130462', '32580',  '149187', '6374', '189323', '453981', '63649', '78844', '375865', '457451', '452034', '277954', '368133', '187336', '7701', '373838', '60387', '384528', '159319', '61458', '143571', '181873', '138298', '162575', '440181', '435891', '348960', '449994', '224609', '358441', '449904', '31126', '66410', '188294', '121146', '44795', '44214', '32858', '224647', '127611', '328142', '199412', '436902', '89406', '198510', '19941', '133910', '436567', '439851', '441799', '222904', '165306', '438532', '142375', '315846', '172704', '448462']
#    direction_vector = ['143510', '367091', '407806', '135034', '118311', '434508', '156473', '32772', '114033', '399890', '370656', '427293', '438786', '11376', '390638',  '175740', '109691', '137247', '165760', '88687', '77787', '235121', '232578', '47273', '426723', '146374', '199481', '5709', '85126', '19776', '110004', '272599', '197926', '196630', '79964', '79478', '72835', '136632', '283878', '87764', '461014', '77170', '436708', '124427', '302737', '138522', '130892', '430400', '285895', '179139', '247988', '113481', '150935', '329799', '130462', '32580',  '149187', '6374', '189323', '453981', '63649', '78844', '375865', '457451', '452034', '277954', '368133', '187336', '7701', '373838', '60387', '384528', '159319', '61458', '143571', '181873', '138298', '162575', '440181', '435891', '348960', '449994', '224609', '358441', '449904', '31126', '66410', '188294', '121146', '44795', '44214', '32858', '224647', '127611', '328142', '199412', '436902', '89406', '198510', '19941', '133910', '436567', '439851', '441799', '222904', '165306', '438532', '142375', '315846', '172704', '448462']