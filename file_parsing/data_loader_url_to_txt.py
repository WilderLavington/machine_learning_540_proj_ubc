import pickle

import numpy as np
import sklearn
import scipy
from sklearn import datasets
import requests


""" LOAD ALL THE TXT FILES IN LIBSVM """

# data set 1 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a1a.txt', 'wb') as f:
    f.write(r.content)

# data set 2 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a2a.txt', 'wb') as f:
    f.write(r.content)

# data set 3 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a3a.txt', 'wb') as f:
    f.write(r.content)

# data set 4 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a4a.txt', 'wb') as f:
    f.write(r.content)

# data set 5 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a5a.txt', 'wb') as f:
    f.write(r.content)

# data set 6 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a6a.txt', 'wb') as f:
    f.write(r.content)

# data set 7 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a7a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a7a.txt', 'wb') as f:
    f.write(r.content)

# data set 8 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a8a.txt', 'wb') as f:
    f.write(r.content)

# data set 9 training data set - test set downloaded manually at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/a9a.txt', 'wb') as f:
    f.write(r.content)

# data set 10 training data set
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/australian_scale.txt', 'wb') as f:
    f.write(r.content)

# data set 11 training data set
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/breast_cancer_scale.txt', 'wb') as f:
    f.write(r.content)

# data set 12 training data set, test set downloaded manually
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/cod_rna.txt', 'wb') as f:
    f.write(r.content)
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/cod_rna.r.txt', 'wb') as f:
    f.write(r.content)

# dataset 13
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/diabetes_scale.txt', 'wb') as f:
    f.write(r.content)


# dataset 14
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/fourclass_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/fourclass_scale.txt', 'wb') as f:
    f.write(r.content)

# dataset 15
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/german.numer_scale.txt', 'wb') as f:
    f.write(r.content)

# dataset 16
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/heart_scale.txt', 'wb') as f:
    f.write(r.content)

# dataset 20
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ionosphere_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/ionosphere_scale.txt', 'wb') as f:
    f.write(r.content)

# dataset 22
example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/liver-disorders_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/liver_disorders_scale.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/madelon.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/mushrooms.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/phishing.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/skin_nonskin'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/skin_nonskin.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/splice_scale.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/sonar_scale'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/sonar_scale.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/svmguide1.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide3'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/svmguide3.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w1a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w1a.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w2a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w2a.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w3a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w3a.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w4a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w4a.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w5a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w5a.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w6a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w6a.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w7a.txt', 'wb') as f:
    f.write(r.content)

example_txt = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a'
r = requests.get(example_txt)
with open('/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/w7a.txt', 'wb') as f:
    f.write(r.content)
