using Random
using LinearAlgebra
using Statistics
using Printf
include("helper_fxns.jl")
include("smo.jl")

# @printf("Iteration: %d\n",count_)
# trainErr[count_] = 0.5*alpha'*(H)*alpha - sum(alpha)
# @printf("Objective Function: %.3f\n", trainErr[count_])
# testErrRate = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]
# testErr[count_] = testErrRate
# @printf("Testing error: %.3f\n", testErr[count_])

# Min over block
function gsq_rule(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w_old, b_old)
    # init minimum
    min_val = Inf

    # init new alpha updates
    alpha_i, alpha_j = 0, 0
    best_block = [1,1]

    # randomize block order
    eval_order = shuffle(collect(1:number_of_blocks))

    # compute gradient
    g = H * alpha - ones(size(y))

    # iterate through blocks
    for i = 1:number_of_blocks
        # pick blocks in random order
        current_block = blocks[eval_order[i],:]
        # evaluate SMO rule
        obj_val, a_i, a_j = smo_block(current_block, alpha, X, y, C, H, g, kernel, w_old, b_old)
        # check if we need to update
        if min_val > obj_val
            # if so, replace the alpha parameters
            alpha_i, alpha_j = a_i, a_j
            # update min value found
            min_val = obj_val
            # best block
            best_block = current_block
        end
    end
    #print(best_block)
    return best_block, alpha_i, alpha_j
end

# Fit function
function fit_gsq_exact(X, y, X_test, y_test, kernel, C, epsilon, max_iter, print_info_)

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
    number_of_blocks, _ = size(blocks)

    # Evaluation
    trainErr[count_] = 0.5*alpha'*(H)*alpha - sum(alpha)
    testErrRate = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]
    testErr[count_] = testErrRate

    # print info
    if print_info_
        print_info(count_, trainErr[count_], testErr[count_])
    end

    # primary loop
    while true

        # update stopping conditions
        count_ += 1

        # compute best block
        best_block, alpha_i, alpha_j = gsq_rule(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w, b)

        # Set new alpha values
        alpha[Int(best_block[1])] = alpha_i
        alpha[Int(best_block[2])] = alpha_j

        # re-compute model parameters
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
        if print_info_
            println(best_block)
            print_info(count_, trainErr[count_], testErr[count_])
        end

        # evaluate KKT conditions
        satified = KKT_conditions(X,y,n,alpha,w,b)

        # stopping condistions
        if satified
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
    # return it all
    return 1, 1, count_, support_vectors, w, b
end
