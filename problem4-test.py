# -*- coding: utf-8 -*-
import numpy as np
import re
from sklearn.externals import joblib
import sys

openpath=sys.argv[1]
fid = open(openpath)
lines = fid.readlines()

f = []

for line in lines:
    line = line.strip('\n')
    line = line.lstrip(' ')
    line_split = re.split('\s+',line)#remove space between line
    line_split = line_split[:257]
    digit_num = float(line_split[0])
    line_split = [float(x) for x in line_split]
    f.append(line_split)


f = np.array(f)


col_array = np.arange(f.shape[1])
x_test = f[:,col_array[1:257]]
y_test = f[:,0]



input = 'D://p4train_model.pkl'
clf = joblib.load(input)
predictions = clf.predict(x_test)
accuracy = clf.score(x_test,y_test)
print('accuracy: %s' % accuracy)
print('test-set error: %s' %clf.loss_)
