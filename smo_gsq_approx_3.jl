using Random
using LinearAlgebra
using Statistics
using Printf
include("helper_fxns.jl")
include("smo.jl")

function compute_diff(grad, L)
    n = length(grad)
    grad_diff = zeros(n, n)
    for i = 1:n
        for j = 1:n
            grad_diff[i,j] = (grad[i] - grad[j])^2 / (L[i] + L[j])
        end
    end
    return grad_diff
end

# Max grad over block
function approx_gsq_rule_3(blocks, number_of_blocks, alpha, X, y, C, H, L, kernel, w_old, b_old)

    # init minimum
    n = size(y)[1]

    # compute gradient
    g = H * alpha - ones(n)

    # pick the first coordinate by largest gradient not equal to
    viable_indices = findall(((g .> 0) .& (alpha .== C)) .| ((g .< 0) .& (alpha .== 0.)) .| (((alpha .< C) .& (alpha .> 0.)) .& (g .!= 0.)))

    # set the values we can actually update
    viable_g = g[viable_indices]
    viable_alpha = alpha[viable_indices]
    viable_L = L[viable_indices]

    # find difference
    diff_mat = compute_diff(viable_g, viable_L)

    # get max coordinates
    coords = findall(diff_mat .== maximum(diff_mat))

    # shuffle
    eval_order = shuffle(collect(1:length(coords)))

    # grab coord indexes
    coord_1 = viable_indices[coords[eval_order[1]][1]]
    coord_2 = viable_indices[coords[eval_order[1]][2]]

    # set this value as our block
    best_block = [coord_1, coord_2]

    # compute the exact update
    alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H,  g, kernel, w_old, b_old)

    # return info
    return best_block, alpha_i, alpha_j
end

# Fit function
function fit_gsq_approx_3(X, y, X_test, y_test, kernel, C, epsilon, max_iter, print_info_)

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

    # compute diag of quadratic
    L = diag(diagnolize(H))

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
        best_block, alpha_i, alpha_j = approx_gsq_rule_3(blocks, number_of_blocks, alpha, X, y, C, H, L, kernel, w, b)

        # compute stopping flag
        stop_flag = norm([alpha_i alpha_j] - [alpha[Int(best_block[1])] alpha[Int(best_block[2])]], 1)

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
            print_info(count_, trainErr[count_], testErr[count_])
        end

        # stopping condistions

        satified, testErr, trainErr = stopping_conditions(testErr,trainErr,X,y,n,alpha,w,b,count_,max_iter,C,epsilon,0)

        # check if we should stop
        if satified
            break
        end
    end

    # compute parameters
    sv = findall((alpha .> 0) .& (alpha .< C))
    w = transpose(X) * (alpha.*y)
    if length(sv) > 0
        b = transpose(w) * X[sv[1],:] - y[sv[1]]
    else
        b = 0
    end

    # return it all
    return trainErr, testErr, count_, sv, w, b
end
