using Random
using LinearAlgebra
using Statistics
using Printf
include("helper_fxns.jl")
include("smo.jl")

# Max grad over block
function approx_gsq_rule_2_a(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w_old, b_old)
    # init minimum
    n = size(y)[1]
    # compute gradient
    g = H * alpha - ones(n)
    # pick the first coordinate by largest gradient not equal to
    viable_indices = findall(((g .> 0) .& (alpha .>= C)) .| ((g .< 0) .& (alpha .<= 0.)) .| ((alpha .< C) .& (alpha .> 0.)))
    # set the values we can actually update
    viable_g = g[viable_indices]
    viable_alpha = alpha[viable_indices]
    # get max coordinates
    coord_1 = findall(viable_g .== maximum(viable_g))
    # check that there are not multiples
    if length(coord_1) >= 1
        eval_order = shuffle(collect(1:length(coord_1)))
        coord_1 = coord_1[eval_order[1]]
    end
    # init new alpha updates
    best_block = [coord_1, 1]
    min_val = Inf
    # iterate through blocks
    for i = 1:n
        # pick blocks in random order
        current_block = [coord_1,i]
        # evaluate SMO rule
        obj_val, _, _ = smo_block(current_block, alpha, X, y, C, H, g, kernel, w_old, b_old)
        # check if we need to update
        if min_val > obj_val
            # update min value found
            min_val = obj_val
            # best block
            best_block = current_block
        end
    end
    # compute the exact update
    _, alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H,  g, kernel, w_old, b_old)
    # return info
    return best_block, alpha_i, alpha_j
end

# Max grad over block
function approx_gsq_rule_2_b(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w_old, b_old)
    # init minimum
    n = size(y)[1]
    # compute gradient
    g = H * alpha - ones(n)
    # pick the first coordinate by largest gradient not equal to
    viable_indices = findall(((g .> 0) .& (alpha .>= C)) .| ((g .< 0) .& (alpha .<= 0.)) .| ((alpha .< C) .& (alpha .> 0.)))
    # set the values we can actually update
    viable_g = g[viable_indices]
    viable_alpha = alpha[viable_indices]
    # get max coordinates
    coord_1 = findall(viable_g .== minimum(viable_g))
    # check that there are not multiples
    if length(coord_1) >= 1
        eval_order = shuffle(collect(1:length(coord_1)))
        coord_1 = coord_1[eval_order[1]]
    end
    # init new alpha updates
    best_block = [coord_1, 1]
    min_val = Inf
    # iterate through blocks
    for i = 1:n
        # pick blocks in random order
        current_block = [coord_1,i]
        # evaluate SMO rule
        obj_val, _, _ = smo_block(current_block, alpha, X, y, C, H, g, kernel, w_old, b_old)
        # check if we need to update
        if min_val > obj_val
            # update min value found
            min_val = obj_val
            # best block
            best_block = current_block
        end
    end
    # compute the exact update
    _, alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H,  g, kernel, w_old, b_old)
    # return info
    return best_block, alpha_i, alpha_j
end

# Fit function
function fit_gsq_approx_2_a(X, y, X_test, y_test, kernel, C, epsilon, max_iter, print_info_)

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
        best_block, alpha_i, alpha_j = approx_gsq_rule_2_a(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w, b)

        # Set new alpha values
        alpha[Int(best_block[1])] = alpha_i
        alpha[Int(best_block[2])] = alpha_j

        # compute best block
        best_block, alpha_i, alpha_j = approx_gsq_rule_2_b(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w, b)

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

        # evaluate KKT conditions
        satified = KKT_conditions(X,y,n,alpha,w,b)

        # stopping condistions
        if satified
            println("KKT conditions satified")
            break
        elseif count_ >= max_iter
            println("exceeded max iterations")
            return trainErr, testErr, count_, sv, w, b
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
    return trainErr, testErr, count_, support_vectors, w, b
end
