using Random
using LinearAlgebra
using Statistics
using Printf
include("smo.jl")
include("helper_fxns.jl")

# get random integer other then current
function resrndint(a, b, z)
    i = z
    count = 0
    vals = randperm(b-a)
    for i = 1:(b-a)
        if vals[i+a] != z
            return vals[i+a]
        end
    end
    print("error in ur codes ~ 26")
    return 0
end

# fit function
function fit_gsq_random(X, y, X_test, y_test, kernel, C, epsilon, max_iter)
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
    testErrRate = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]
    testErr[count_] = testErrRate

    # print info
    print_info(count_, trainErr[count_], testErr[count_])

    # primary loop
    while true
        count_ += 1

        # get random integer between 0, and n-1 != j
        j = rand(1:n)
        i = resrndint(0, n-1, j)

        # compute gradient
        g = (y * y').*(X * X') * alpha - ones(size(y))

        # evaluate SMO rule
        obj_val, a_i, a_j = smo_block([i,j], alpha, X, y, C, H, g, kernel, w, b)

        # update alphas
        alpha[i] = a_i
        alpha[j] = a_j

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
        testErrRate = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]
        testErr[count_] = testErrRate

        # print info
        print_info(count_, trainErr[count_], testErr[count_])

        # evaluate KKT conditions
        satified = KKT_conditions(X,y,n,alpha,w,b)

        # stopping condistions
        if satified
            testErr[count_:end] .= testErr[count_]
            trainErr[count_:end] .= trainErr[count_]
            println("KKT conditions satified")
            break
        elseif count_ >= max_iter
            println("exceeded max iterations")
            break
        end
    end
    # find a support vector
    sv = findall((alpha .> 0) .& (alpha .< C))[1]
    # Compute model parameters
    w = transpose(X) * (alpha.*y)
    b = transpose(w) * X[sv,:] - y[sv]
    # Get support vectors
    alpha_idx = findall((alpha .> 0) .& (alpha .< C))
    support_vectors = X[alpha_idx, :]
    return trainErr, testErr, count_, support_vectors
end
