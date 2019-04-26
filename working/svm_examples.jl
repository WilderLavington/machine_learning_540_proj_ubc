
include("./smo_gsq.jl")
include("./random_smo.jl")

# import data
using RDatasets, LIBSVM
using Random
using LinearAlgebra
using Statistics


a2a = dataset("datasets", "australian")
a2a = dataset("datasets", "a2a")

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")
# LIBSVM handles multi-class data automatically using a one-against-one strategy
labels = iris[:Species]
# First dimension of input data is features; second is instances
instances = iris[:, 1:4]

# set training data
X_train = convert(Matrix, instances[1:2:end, :])
y_train = labels[1:2:end]

# set test data
X_test = convert(Matrix, instances[2:2:end, :])
y_test =  labels[2:2:end]

""" BINARY CLASSIFICATION - REAL DATA"""

# binary labels
label_1train = findall(y_train .==  "versicolor")
label_2train = findall(y_train .==  "setosa")
label_1test = findall(y_test .==  "versicolor")
label_2test = findall(y_test .==  "setosa")
binary_labels_train = [label_1train;label_2train]
binary_labels_test = [label_1test;label_2test]

# set training data
X_train_bin = X_train[binary_labels_train, :]
y_train_bin = zeros(length(binary_labels_train))
y_train_bin[label_1train] .= 1
y_train_bin[label_2train] .= -1

# set test data
X_test_bin = X_test[binary_labels_test, :]
y_test_bin =  zeros(length(binary_labels_test))
y_test_bin[label_1test] .= 1
y_test_bin[label_2test] .= -1

# hyper parameters
max_iter = 10000
C = 1.0
epsilon = 0.001

# train model
support_vectors, count, w, b = fit_gsq(X, y, kernel, C, epsilon, max_iter)
# look at test error
pred = predict(X_test_bin, w, b)
print(sum((pred .!= y_test_bin)))

# train model
support_vectors, count, w, b = randomfit(X, y, kernel, C, epsilon, max_iter)
# look at test error
pred = predict(X_test_bin, w, b)
print(sum((pred .!= y_test_bin)))


""" BINARY CLASSIFICATION - FAKE DATA"""

X_fake = rand(100,2)
X_fake[1:50,:] = X_fake[1:50,:] - 2*rand(50,2)
y_fake = ones(100)
y_fake[1:50] = -1*ones(50)

# hyper parameters
max_iter = 10000
C = 1.0
epsilon = 0.001

# train model
support_vectors, count, w, b = fit_gsq(X, y, kernel, C, epsilon, max_iter)
# look at test error
pred = predict(X_test_bin, w, b)
print(sum((pred .!= y_test_bin)))

# train model
support_vectors, count, w, b = randomfit(X, y, kernel, C, epsilon, max_iter)
# look at test error
pred = predict(X_test_bin, w, b)
print(sum((pred .!= y_test_bin)))
