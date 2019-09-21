# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:49:26 2019

@author: y374zhou
"""

from __future__ import print_function
#import sys


import numpy as np

try:
	import scipy
	from scipy import sparse
except:
	scipy = None
	sparse = None


#__all__ = ['svm_read_problem', 'evaluations', 'csr_find_scale_param', 'csr_scale']

def svm_read_problem(data_file_name, return_scipy=False):
	"""
	svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
	svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix
	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	row_ptr = [0]
	col_idx = []
	for i, line in enumerate(open(data_file_name)):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		prob_y += [float(label)]
		if scipy != None and return_scipy:
			nz = 0
			for e in features.split():
				ind, val = e.split(":")
				val = float(val)
				if val != 0:
					col_idx += [int(ind)-1]
					prob_x += [val]
					nz += 1
			row_ptr += [row_ptr[-1]+nz]
		else:
			xi = {}
			for e in features.split():
				ind, val = e.split(":")
				xi[int(ind)] = float(val)
			prob_x += [xi]
	if scipy != None and return_scipy:
		prob_y = scipy.array(prob_y)
		prob_x = scipy.array(prob_x)
		col_idx = scipy.array(col_idx)
		row_ptr = scipy.array(row_ptr)
		prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))
	return (prob_y, prob_x)

y, x = svm_read_problem(r'./heart', return_scipy=True)
x = x.toarray()
data = np.append(np.array(y).reshape(-1,1), np.array(x), axis=1)
np.savetxt('heart.csv', data, delimiter=',')
