
include("./random_smo.jl")
include("./smo_gsq_exact.jl")
include("./smo_gsq_approx_1.jl")
# include("./smo_efficient_gsq_j.jl")
# include("./smo_efficient_gsq_i.jl")
# include("./smo_efficient_gsq_alternij.jl")
# include("./smo_gsq_identityHess.jl")
# include("./smo_gsq_diagApprx.jl")
# include("./smo_gsq_diagApprxH.jl")
# include("./smo_gss.jl")
# import data
using Random
using LinearAlgebra
using Statistics

# function generate_data(data_size, data_dims, split, centroid_1, centroid_2, scaling)
#     X_fake = rand(data_size, data_dims)
#     plus = floor(data_size*split)
#
#
# end

X_fake = rand(10,2)
X_fake[1:5,:] = X_fake[1:5,:] - 2*rand(5,2)
y_fake = ones(10)
y_fake[1:5] = -1*ones(5)


X_faket = rand(10,2)
X_faket[1:5,:] = X_faket[1:5,:] - 2*rand(5,2)
y_faket = ones(10)
y_faket[1:5] = -1*ones(5)

# hyper parameters
max_iter = 1e4
C = 1.0
epsilon = 0.01
kernal_func = linear_kernal

# random
trainErr_1, testErr_1, count_1, support_vectors_1 = fit_gsq_random(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# exact
trainErr_2, testErr_2, count_2, support_vectors_2 = fit_gsq_exact(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)
# first approximation H = L * I
trainErr_3, testErr_3, count_3, support_vectors_3 = fit_gsq_approx_1(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter)

# # now train using libsvm
# using LIBSVM
# model = svmtrain(X_fake', y_fake, verbose=true)
# (predicted_labels, decision_values) = svmpredict(model, X_fake')

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
