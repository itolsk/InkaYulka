import numpy as np
import pandas as pd
import csv
import string
import nltk
from nltk.corpus import names
from unidecode import unidecode

df = pd.read_csv('/Users/ito022/Downloads/ru_train.csv', delimiter=',', lineterminator='\n', error_bad_lines=False, low_memory=False, encoding = 'utf8',  names=['sentence_id', 'token_id', 'class', 'before', 'after'])

classes=df['class'].tolist()
before=df['before'].tolist()
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
def features(input_string, word_features):
    features = {}
    features['is float']=isinstance(input_string, float)
    features['contains roman numbers']=isinstance(input_string, str) and all((c in ('IVXLNM')) for c in input_string)
    features['contains dates']=isinstance(input_string, str) and any((c in ('год','январ','феврал', 'март','апрел','мая','июн','июл','август','сентябр','октябр','ноябр', 'декабр')) for c in input_string)
    features['year']=isinstance(input_string, float) and (input_string<3000)and (input_string>1000)
    features['contains numbers']=isinstance(input_string, str) and any(char.isdigit() for char in input_string)
    features['contains /']=isinstance(input_string, str) and '/' in input_string
    features['contains punctuation']=isinstance(input_string, str) and any(char in string.punctuation for char in input_string)
    #features['number']=''.join[char for char in input_string.split() if char.isdigit()]                                                                                                                                                                                                                                  
    features['short'] = isinstance(input_string, str) and len(input_string)<4
    features['contains capital latin']=isinstance(input_string, str) and re.search('[A-Z]', input_string)==None
    features['contains lowercase latin']=isinstance(input_string, str) and re.search('[a-z]', input_string)==None
    features['contains capital cyrillic']=isinstance(input_string, str) and re.search('[А-Я]', input_string)==None
    features['contains lowercase cyrillic']=isinstance(input_string, str) and re.search('[а-я]', input_string)==None
    features['contains upper']=isinstance(input_string, str) and any(c.isupper() for c in input_string)    
    features['contains lower']=isinstance(input_string, str) and any(c.islower() for c in input_string) 
    features['contains measures']=isinstance(input_string, str) and any((c in ('st', 'мин', 'с.', 'км', 'см', '%','метр','л.','В','гб','гр','грам','кило','га', 'тыс','ярд','А','мм','В', 'тонн')) for c in input_string) and any(char.isdigit() for char in input_string)
    features['contains endings']=isinstance(input_string, str) and any((c in ('-го', '-ом','-й','-е','-я','-и','-х', '—')) for c in input_string)
    features['contains —']=isinstance(input_string, str) and '—' in input_string
    features['contains /']=isinstance(input_string, str) and '/' in input_string
    features['contains і']=isinstance(input_string, str) and 'і' in input_string
    features['contains ї']=isinstance(input_string, str) and 'ї' in input_string
    features['contains &']=isinstance(input_string, str) and '&' in input_string
    features['contains -']=isinstance(input_string, str) and '-' in input_string
    features['is -']=isinstance(input_string, str) and input_string=='-'
      
    return features

words=set(before)
labeled_names=zip(before,classes)
features_n = [features(n, words) for n in before]
featuresets = list(zip(features_n, classes))
train_set, test_set = featuresets[4500:], featuresets[:500]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print('classifier accuracy', nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(25)



#################
train_names = labeled_names_to_list[1500:]
devtest_names = labeled_names_to_list[500:1500]
test_names = labeled_names_to_list[:500]
train_set = featuresets_to_list[1500:]
devtest_set = featuresets_to_list[500:1500]
test_set = featuresets_to_list[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set) 
print(nltk.classify.accuracy(classifier, test_set))

errors = []
for (name, tag) in devtest_names:
     guess = classifier.classify(features(name, words))
     if guess != tag:
        errors.append( (tag, guess, name) )

for (tag, guess, name) in sorted(errors):
    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))
