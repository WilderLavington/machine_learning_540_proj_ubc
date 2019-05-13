include("./smo_gsq.jl")
include("./random_smo.jl")
include("./smo_efficient_gsq_j.jl")
include("./smo_efficient_gsq_i.jl")
include("./smo_efficient_gsq_alternij.jl")
include("./smo_gsq_identityHess.jl")
include("./smo_gsq_diagApprx.jl")
include("./smo_gsq_diagApprxH.jl")
include("./smo_gss.jl")
# import data
using RDatasets, LIBSVM
using Random
using LinearAlgebra
using Statistics
using PyPlot
plt = PyPlot

# """ BINARY CLASSIFICATION - REAL DATA"""
#
# # Load Fisher's classic iris data
# iris = dataset("datasets", "iris")
# # LIBSVM handles multi-class data automatically using a one-against-one strategy
# labels = iris[:Species]
# # First dimension of input data is features; second is instances
# instances = iris[:, 1:4]
#
# # set training data
# X_train = convert(Matrix, instances[1:2:end, :])
# y_train = labels[1:2:end]
#
# # set test data
# X_test = convert(Matrix, instances[2:2:end, :])
# y_test =  labels[2:2:end]
#
# # binary labels
# label_1train = findall(y_train .==  "versicolor")
# label_2train = findall(y_train .==  "setosa")
# label_1test = findall(y_test .==  "versicolor")
# label_2test = findall(y_test .==  "setosa")
# binary_labels_train = [label_1train;label_2train]
# binary_labels_test = [label_1test;label_2test]
#
# # set training data
# X_train_bin = X_train[binary_labels_train, :]
# y_train_bin = zeros(length(binary_labels_train))
# y_train_bin[label_1train] .= 1
# y_train_bin[label_2train] .= -1
#
# # set test data
# X_test_bin = X_test[binary_labels_test, :]
# y_test_bin =  zeros(length(binary_labels_test))
# y_test_bin[label_1test] .= 1
# y_test_bin[label_2test] .= -1
#
# # hyper parameters
# max_iter = 1e6
# C = 1.0
# epsilon = 0.001
# kernal_func = linear_kernal
# x = collect(1:max_iter)
#
#
# # train
# trainErr_1, testErr_1, count_1, support_vectors_1 = randomfit(X_train_bin, y_train_bin, X_test_bin, y_test_bin, kernal_func, C, epsilon, max_iter)
# trainErr_2, testErr_2, count_2, support_vectors_2 = fit_gsq(X_train_bin, y_train_bin, X_test_bin, y_test_bin, kernal_func, C, epsilon, max_iter)
# trainErr_3, testErr_3, count_3, support_vectors_3 = fit_approx_gsq(X_train_bin, y_train_bin, X_test_bin, y_test_bin, kernal_func, C, epsilon, max_iter)
#
#
# close()
# figure(1)
# plt.plot(x[1:count_1], trainErr_1[1:count_1], label = "SMO - Uniform Random")
# plt.plot(x[1:count_1], trainErr_2[1:count_1], label = "SMO - GS-q Rule")
# plt.title("Iris Dataset")
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Dual Objective")
# plt.savefig("fig1.png")
# gcf()
#
# figure(2)
# plt.plot(x[1:count_2+25], trainErr_1[1:count_2+25], label = "SMO - Uniform Random")
# plt.plot(x[1:count_2], trainErr_2[1:count_2], label = "SMO - GS-q Rule")
# plt.title("Iris Dataset [Zoomed]")
# plt.xlim(1, count_2+25)
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Dual Objective")
# plt.savefig("fig2.png")
# gcf()

""" BINARY CLASSIFICATION - FAKE DATA"""

X_fake = rand(100,2)
X_fake[1:50,:] = X_fake[1:50,:] - 2*rand(50,2)
y_fake = ones(100)
y_fake[1:50] = -1*ones(50)

X_faket = rand(100,2)
X_faket[1:50,:] = X_faket[1:50,:] - 2*rand(50,2)
y_faket = ones(100)
y_faket[1:50] = -1*ones(50)

# hyper parameters
max_iter = 1e6
C = 1.0
epsilon = 0.005
kernal_func = linear_kernal

# random
trainErr_1, testErr_1, count_1, support_vectors_1 = randomfit(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# exact
trainErr_2, testErr_2, count_2, support_vectors_2 = fit_gsq(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# approximate iterations through all idxs
trainErr_3, testErr_3, count_3, support_vectors_3 = fit_approx_gsq_i(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# trainErr_4, testErr_4, count_4, support_vectors_4 = fit_approx_gsq_j(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
trainErr_5, testErr_5, count_5, support_vectors_5 = fit_approx_gsq_alternij(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# hessian approximations
trainErr_6, testErr_6, count_6, support_vectors_6 = fit_gsq_identityHess(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# psuedocode approximation of gs-q with L*I
trainErr_7, testErr_7, count_7, support_vectors_7 = fit_gsq_diagApprx(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# psuedocode approximation of gs-q with Diag(H)
trainErr_8, testErr_8, count_8, support_vectors_8 = fit_gsq_diagApprxH(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# psuedocode approximation of gs-q with Diag(H)
trainErr_8, testErr_8, count_8, support_vectors_8 = fit_gss(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
#

println(count_1)
println(count_2)
println(count_3)
# println(count_4)
println(count_5)
println(count_6)
# println(count_7)

# now train using libsvm
using LIBSVM
model = svmtrain(X_fake', y_fake, verbose=true)
(predicted_labels, decision_values) = svmpredict(model, X_fake')

# close()
# figure(3)
# plt.plot(x[1:count_1], trainErr_1[1:count_1], label = "SMO - Uniform Random")
# plt.plot(x[1:count_1], trainErr_2[1:count_1], label = "SMO - GS-q Rule")
# plt.title("Synthetic Dataset")
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Dual Objective")
# plt.savefig("fig3.png")
# gcf()
#
# figure(4)
# plt.plot(x[1:count_2+100], trainErr_1[1:count_2+100], label = "SMO - Uniform Random")
# plt.plot(x[1:count_2], trainErr_2[1:count_2], label = "SMO - GS-q Rule")
# plt.title("Synthetic Dataset [Zoomed]")
# plt.xlim(1, count_2 + 100)
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Dual Objective")
# plt.savefig("fig4.png")
# gcf()
