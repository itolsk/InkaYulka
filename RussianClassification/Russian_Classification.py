import numpy as np
import pandas as pd
import csv
import os.path
import string
import nltk
import collections
from collections import Counter
from collections import namedtuple
from nltk.corpus import names
import unidecode
from unidecode import unidecode
import random

df = pd.read_csv('/Users/ito022/Downloads/ru_train.csv', delimiter=',', lineterminator='\n', error_bad_lines=False, low_memory=False, encoding = 'utf8',  names=['sentence_id', 'token_id', 'class', 'before', 'after'])

classes=df['class'].dropna().tolist()

classes_set= set(classes)
#Out[63]: 
#{'CARDINAL',
# 'DATE',
# 'DECIMAL',
# 'DIGIT',
# 'ELECTRONIC',
# 'FRACTION',
# 'LETTERS',
# 'MEASURE',
# 'MONEY',
# 'ORDINAL',
# 'PLAIN',
# 'PUNCT',
# 'TELEPHONE',
# 'TIME',
# 'VERBATIM',


#def features(input_string):
 #   features = {}
  #  features['contains numbers']=any(char.isdigit() for char in input_string)
   # features['contains punctuation']=any(char in string.punctuation for char in input_string) 
    #features['number']=''.join[char for char in input_string.split() if char.isdigit()]
    #features['length'] = len(input_string))
    #return features

def features(input_string, word_features): 
    features = {}
    for word in word_features:
        features['contains({})'.format(str(word).encode('utf-8'))] = (str(word).encode('utf-8') in input_string)
    features['contains numbers']=any(char.isdigit() for char in input_string)
    features['contains punctuation']=any(char in string.punctuation for char in input_string) 
    #features['number']=''.join[char for char in input_string.split() if char.isdigit()]
    features['length'] = len(input_string) 
    return features

words=set(df['before'])
subset = df[['before', 'class']]
labeled_names = [(x, cl) for x, cl in subset.values]
random.shuffle(labeled_names)
featuresets = [(features(n, words), cl) for (n, cl) in labeled_names]
train_set, test_set = featuresets[3000:], featuresets[:3000]
classifier = nltk.NaiveBayesClassifier.train(train_set)




print('classifier accuracy', nltk.classify.accuracy(classifier, test_set))
