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
function approx_gsq_rule_4(blocks, number_of_blocks, alpha, X, y, C, H, approx_H, kernel, w_old, b_old)

    # init minimum
    n, d = size(X)

    # randomize block order
    eval_order = shuffle(collect(1:number_of_blocks))

    # compute gradient
    g = H * alpha - ones(size(y))

    # get a random block as starting point
    best_block = blocks[eval_order[1],:]
    i, j = Int(best_block[1]),Int(best_block[2])

    # get the updates from this point
    alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, approx_H, g, kernel, w_old, b_old)

    # get gradient
    g_b = g[[i, j]]
    # compute d
    d_b = [alpha_i-alpha[i], alpha_j-alpha[j]]
    # compute H
    H_b = [approx_H[i, i] approx_H[i, j]; approx_H[j, i] approx_H[j, j]]
    # get value
    min_val = g_b'*d_b + (d_b'*H_b*d_b) / 2

    # iterate through blocks
    for i = 2:number_of_blocks
        # pick blocks in random order
        current_block = blocks[eval_order[i],:]
        i, j = Int(current_block[1]),Int(current_block[2])

        # evaluate SMO rule
        alpha_prime_i, alpha_prime_j = smo_block(current_block, alpha, X, y, C, approx_H, g, kernel, w_old, b_old)

        # get gradient
        g_b = g[[i, j]]
        # compute d
        d_b = [alpha_prime_i-alpha[i], alpha_prime_j-alpha[j]]
        # compute H
        H_b = [approx_H[i, i] approx_H[i, j]; approx_H[j, i] approx_H[j, j]]
        # get value
        obj_val = g_b'*d_b + (d_b'*H_b*d_b) / 2

        # check if we need to update
        if (min_val > obj_val) & (d_b'*d_b > 0.)
            # update min value found
            min_val = obj_val
            # best block
            best_block = current_block
        end
    end

    # compute the exact update
    alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H,  g, kernel, w_old, b_old)

    # set stopping flag
    stop_flag = 1*(min_val == 0.)

    # return info
    return best_block, alpha_i, alpha_j, 0
end

# Fit function
function fit_gsq_approx_4(X, y, X_test, y_test, kernel, C, epsilon, max_iter, print_info_)

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

    # pre_compute approx hessian
    # approx_H =  diagm(map(x->maximum(H[x,:]), [i for i = 1:size(H)[1]]))
    approx_H = 2 * Diagonal(H)

    # pre-compute number of blocks
    number_of_blocks, _ = size(blocks)

    # Evaluation
    trainErr[count_] = 0.5*alpha'*(H)*alpha - sum(alpha)
    testErr[count_] = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]

    # print info
    if print_info_
        print_info(count_, trainErr[count_], testErr[count_])
    end

    # primary loop
    while true

        # update stopping conditions
        count_ += 1

        # compute best block
        best_block, alpha_i, alpha_j, stop_flag = approx_gsq_rule_4(blocks, number_of_blocks, alpha, X, y, C, H, approx_H, kernel, w, b)
        # println(best_block)
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
        testErr[count_] = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]

        # print info
        if print_info_
            print_info(count_, trainErr[count_], testErr[count_])
        end

        # stopping condistions
        satified, testErr, trainErr, alpha = stopping_conditions(testErr,trainErr,X,y,n,alpha,w,b,count_,max_iter,C,epsilon,stop_flag)

        # check if we should stop
        if satified
            break
        end
    end
    # Compute model parameters
    sv = findall((alpha .> 0) .& (alpha .< C))
    # return it all
    return trainErr, testErr, count_, sv
end
