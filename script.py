'''
FWAF - Machine Learning driven Web Application Firewall
Author: Faizan Ahmad
Website: http://fsecurify.com
'''

import pandas as pd
import numpy as np
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import urllib

def getNGrams(query): #tokenizer function, this will make 3 grams of each query
        tempQuery = str(query)
        ngrams = []
        for i in range(0,len(tempQuery)-3):
                ngrams.append(tempQuery[i:i+3])
        return ngrams

def getQueryFromFile(filename='badqueries.txt'):
        directory = str(os.getcwd())
        filepath = directory + "/" + filename
        data = open(filepath,'r').readlines()
        data = list(set(data))
        queries = set()
        for d in data:
                d = d.strip()
                try:
                        d = str(urllib.unquote(d).decode('utf8'))   #converting url encoded data to simple string
                        queries.add(d)
                except:
                        print 'decode ' + d + ' error'
        return list(queries)



badQueries = getQueryFromFile('badqueries.txt')
tempvalidQueries = getQueryFromFile('goodqueries.txt')
tempAllQueries = badQueries + tempvalidQueries

ybad = np.ones(len(badQueries))
ygood = np.zeros(len(tempvalidQueries))
y = np.hstack((ybad, ygood))

queries = tempAllQueries
vectorizer = TfidfVectorizer(tokenizer=getNGrams) #converting data to vectors
X = vectorizer.fit_transform(queries)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting data

lgs = LogisticRegression()
lgs.fit(X_train, y_train) #training our model
print(lgs.score(X_test, y_test))  #checking the accuracy
