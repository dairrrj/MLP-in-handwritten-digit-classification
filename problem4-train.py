# -*- coding: utf-8 -*-
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import pickle
import sys

openpath=sys.argv[1]
fid = open(openpath)
lines = fid.readlines()

f=[]
for line in lines:
    line = line.strip('\n')
    line = line.lstrip(' ')
    line = line.rstrip(' ')
    line_split= re.split('\s+',line)#remove space between line
    line_split = [float(x) for x in line_split]
    f.append(line_split)


f = np.array(f)

col_array = np.arange(f.shape[1])
xtrain15 = f[:,col_array[1:257]]
ytrain15 = f[:,0]

sfolder = StratifiedKFold(n_splits=3,random_state=3,shuffle=True)
i=0
maxscore=0
for train,test in sfolder.split(xtrain15,ytrain15):
    i=i+1
    print(i)
    x_train = xtrain15[train]
    y_train = ytrain15[train]
    x_test = xtrain15[test]
    y_test = ytrain15[test]
    clf = MLPClassifier(hidden_layer_sizes=[300],activation='logistic',solver='adam',max_iter=200)
    clf.fit(x_train,y_train)
    print('layer size: %s, outputs: %s'%(clf.n_layers_,clf.n_outputs_))
    predictions = clf.predict(x_test)
    accuracy = clf.score(x_test,y_test)
    print('accuracy: %s' % accuracy)
    y_pred = predictions
    print('in-sample error: %s' % clf.loss_)
    if maxscore < accuracy:
        maxscore = accuracy
        s = pickle.dumps(clf)

clf2 = pickle.loads(s)
output = 'D://p4train_model.pkl'
joblib.dump(clf,output)
