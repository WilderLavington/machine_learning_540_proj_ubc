using Random
using LinearAlgebra
using Statistics
using Printf
include("smo.jl")
include("helper_fxns.jl")

# get random integer other then current
function resrndint(b, z)
    i = z
    count = 0
    vals = randperm(b)
    for i = 1:(b)
        if vals[i] != z
            return vals[i]
        end
    end
    print("error in ur codes ~ 18")
    return 0
end

# fit function
function fit_gsq_random(X, y, X_test, y_test, kernel, C, epsilon, max_iter, print_info_)
    # generate all blocks
    blocks = generate_blocks(length(y))
    # Initializations
    n, d = size(X)
    alpha = zeros(n)

    # data storage
    trainErr = zeros(Int(max_iter))
    testErr = zeros(Int(max_iter))
    # count for maxing out iterations
    count_ = 1

    # Compute model parameters
    sv = findall((alpha .> 0) .& (alpha .< C))
    w = transpose(X) * (alpha.*y)
    if length(sv) > 0
        b = transpose(w) * X[sv[1],:] - y[sv[1]]
    else
        b = 0
    end

    # pre-compute hessian
    H = (y * y').*(X * X')
    # pre-compute number of blocks
    number_of_blocks = length(blocks)

    # Evaluation
    trainErr[count_] = 0.5*alpha'*(H)*alpha - sum(alpha)
    testErr[count_] = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]

    # print info
    if print_info_
        print_info(count_, trainErr[count_], testErr[count_])
    end

    # primary loop
    while true
        count_ += 1

        # get random integer between 0, and n-1 != j
        i = rand(1:n)
        j = resrndint(n, i)

        # compute gradient
        g = H * alpha - ones(size(y))

        # evaluate SMO rule
        alpha_prime_i, alpha_prime_j = smo_block([i,j], alpha, X, y, C, H, g, kernel, w, b)

        # update alphas
        alpha[i] = alpha_prime_i
        alpha[j] = alpha_prime_j

        # Compute model parameters
        sv = findall((alpha .> 0) .& (alpha .< C))
        w = transpose(X) * (alpha.*y)
        if length(sv) > 0
            b = transpose(w) * X[sv[1],:] - y[sv[1]]
        else
            b = 0
        end

        # Evaluation
        trainErr[count_] = 0.5*alpha'*(H)*alpha - sum(alpha)
        testErr[count_] = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]

        # print info
        if print_info_
            print_info(count_, trainErr[count_], testErr[count_])
        end

        # stopping condistions
        satified, testErr, trainErr, alpha = stopping_conditions(testErr,trainErr,X,y,n,alpha,w,b,count_,max_iter,C,epsilon,0)

        # check if we should stop
        if satified
            break
        end

    end
    # return
    return trainErr, testErr, count_, alpha
end
