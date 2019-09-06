
import pickle
import numpy as np
import sklearn
import scipy
from sklearn import datasets
import requests

# data set 0 - 9 conversion
for i in range(1,10):
    fname = 'data_sets_pkl/data_set_' + str(i) + '/a' + str(i) + 'a_train'
    f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(i) + '/a' + str(i) + 'a.txt'
    data = datasets.load_svmlight_file(f)
    pickle.dump(data, open( fname + '.pkl', "wb" ) )
    fname = 'data_sets_pkl/data_set_' + str(i) + '/a' + str(i) + 'a_test'
    f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(i) + '/a' + str(i) + 'a.t.txt'
    data = datasets.load_svmlight_file(f)
    pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 10
fname = 'data_sets_pkl/data_set_' + str(10) + '/' + 'a' + str(10) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(10) + '/australian_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 11
fname = 'data_sets_pkl/data_set_' + str(11) + '/' + 'a' + str(11) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(11) + '/breast_cancer_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 12
fname = 'data_sets_pkl/data_set_' + str(12) + '/' + 'a' + str(12) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(12) + '/cod_rna.r.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(12) + '/' + 'a' + str(12) + 'a_valid'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(12) + '/cod_rna.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(12) + '/' + 'a' + str(12) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(12) + '/cod_rna.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 13
fname = 'data_sets_pkl/data_set_' + str(13) + '/' + 'a' + str(13) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(13) + '/colon-cancer.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 14
fname = 'data_sets_pkl/data_set_' + str(14) + '/' + 'a' + str(14) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(14) + '/covtype.libsvm.binary.scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 15
fname = 'data_sets_pkl/data_set_' + str(15) + '/' + 'a' + str(15) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(15) + '/diabetes_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 16
fname = 'data_sets_pkl/data_set_' + str(16) + '/' + 'a' + str(16) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(16) + '/duke.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(16) + '/' + 'a' + str(16) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(16) + '/duke.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 17
fname = 'data_sets_pkl/data_set_' + str(17) + '/' + 'a' + str(17) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(17) + '/fourclass_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 18
fname = 'data_sets_pkl/data_set_' + str(18) + '/' + 'a' + str(18) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(18) + '/german.numer_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 19
fname = 'data_sets_pkl/data_set_' + str(19) + '/' + 'a' + str(19) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(19) + '/heart_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 20
fname = 'data_sets_pkl/data_set_' + str(20) + '/' + 'a' + str(20) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(20) + '/ijcnn1.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(20) + '/' + 'a' + str(20) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(20) + '/ijcnn1.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(20) + '/' + 'a' + str(20) + 'a_valid'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(20) + '/ijcnn1.v.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 21
fname = 'data_sets_pkl/data_set_' + str(21) + '/' + 'a' + str(21) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(21) + '/leu.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(21) + '/' + 'a' + str(21) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(21) + '/leu.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 22
fname = 'data_sets_pkl/data_set_' + str(22) + '/' + 'a' + str(22) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(22) + '/liver-disorders.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(22) + '/' + 'a' + str(22) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(22) + '/liver_disorders_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 23
fname = 'data_sets_pkl/data_set_' + str(23) + '/' + 'a' + str(23) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(23) + '/madelon.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(23) + '/' + 'a' + str(23) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(23) + '/madelon.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 24
fname = 'data_sets_pkl/data_set_' + str(24) + '/' + 'a' + str(24) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(24) + '/mushrooms.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 25
fname = 'data_sets_pkl/data_set_' + str(25) + '/' + 'a' + str(25) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(25) + '/phishing.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 26
fname = 'data_sets_pkl/data_set_' + str(26) + '/' + 'a' + str(26) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(26) + '/news20.binary.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 27
fname = 'data_sets_pkl/data_set_' + str(27) + '/' + 'a' + str(27) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(27) + '/real-sim.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 28
fname = 'data_sets_pkl/data_set_' + str(28) + '/' + 'a' + str(28) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(28) + '/rcv1_test.binary.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(28) + '/' + 'a' + str(28) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(28) + '/rcv1_train.binary.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 29
fname = 'data_sets_pkl/data_set_' + str(29) + '/' + 'a' + str(29) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(29) + '/skin_nonskin.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 30
fname = 'data_sets_pkl/data_set_' + str(30) + '/' + 'a' + str(30) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(30) + '/splice.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(30) + '/' + 'a' + str(30) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(30) + '/splice_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 31
fname = 'data_sets_pkl/data_set_' + str(31) + '/' + 'a' + str(31) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(31) + '/sonar_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 32
fname = 'data_sets_pkl/data_set_' + str(32) + '/' + 'a' + str(32) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(32) + '/SUSY.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 33
fname = 'data_sets_pkl/data_set_' + str(33) + '/' + 'a' + str(33) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(33) + '/svmguide1.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(33) + '/' + 'a' + str(33) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(33) + '/svmguide1.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 34
fname = 'data_sets_pkl/data_set_' + str(34) + '/' + 'a' + str(34) + 'a_test'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(34) + '/svmguide3.t.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
fname = 'data_sets_pkl/data_set_' + str(34) + '/' + 'a' + str(34) + 'a_train'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(34) + '/svmguide3.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )

# # dataset 35
# fname = 'data_sets_pkl/data_set_' + str(35) + '/' + 'a' + str(35) + 'a_all'
# f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(35) + '/url_combined.txt'
# data = datasets.load_svmlight_file(f)
# pickle.dump(data, open( fname + '.pkl', "wb" ) )

# data set 36 - 43 conversion
for i in range(0,9):
    print(i)
    fname = 'data_sets_pkl/data_set_' + str(i) + '/a' + str(36+i) + 'a_train'
    f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(i) + '/w' + str(i) + 'a.txt'
    data = datasets.load_svmlight_file(f)
    pickle.dump(data, open( fname + '.pkl', "wb" ) )
    fname = 'data_sets_pkl/data_set_' + str(i) + '/a' + str(36+i) + 'a_test'
    f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(i) + '/w' + str(i) + 'a.t.txt'
    data = datasets.load_svmlight_file(f)
    pickle.dump(data, open( fname + '.pkl', "wb" ) )

# dataset 44
fname = 'data_sets_pkl/data_set_' + str(44) + '/' + 'a' + str(44) + 'a_all'
f = '/Users/wilder/Desktop/machine_learning_540_proj_ubc-master/python/data_sets_txt/data_set_' + str(44) + '/ionosphere_scale.txt'
data = datasets.load_svmlight_file(f)
pickle.dump(data, open( fname + '.pkl', "wb" ) )
