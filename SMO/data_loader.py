import pickle

import numpy as np
import sklearn
import scipy

fname = ''

data = sklearn.datasets.load_svmlight_file(f)

pickle.dump(data, open( fname + '.pkl', "wb" ) )

# Function for loading pickled datasets
def load_sparse_data(data_dir, data_name, dataset_num):
    # real data
    data = pickle.load(open(data_dir + data_name +'.pkl', 'rb'), encoding = "latin1")

    # load real dataset; it's stored as a sparse matrix so you need to cast to a np array.
    A = data[0].toarray()

    # check if the targets are stored as a sparse matrix
    if isinstance(data[1], scipy.sparse.csr.csr_matrix):
        y = data[1].toarray().ravel()
    else:
        y = data[1]

    # for binary data only: use {-1,1} targets
    y[(np.where(y == np.unique(y)[0]))[0].tolist()] = -1
    y[(np.where(y == np.unique(y)[1]))[0].tolist()] = 1

    return A, y
