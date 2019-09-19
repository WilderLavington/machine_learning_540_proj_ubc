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
    viable_indices = findall(((g .>= 0) .& (alpha .== C)) .| ((g .<= 0) .& (alpha .== 0.)) .| ((alpha .< C) .& (alpha .> 0.)))
    # set the values we can actually update
    viable_g = g[viable_indices]
    viable_alpha = alpha[viable_indices]
    # get max coordinates
    coord_1 = findall(viable_g .== maximum(viable_g))
    # check that there are not multiples
    if length(coord_1) >= 1
        eval_order = shuffle(collect(1:length(coord_1)))
        coord_1 = viable_indices[coord_1[eval_order[1]]]
    end
    # now reshuffle eval order
    eval_order = shuffle(collect(1:n))
    # init new alpha updates
    if all(coord_1 .!= eval_order[1])
        # compute alphas
        best_block = [coord_1, eval_order[1]]
        i, j = best_block
        alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H, g, kernel, w_old, b_old)
        # get gradient
        g_b = g[[i, j]]
        # compute d
        d_b = [alpha_i-alpha[i], alpha_j-alpha[j]]
        # compute H
        H_b = [H[i, i] H[i, j]; H[j, i] H[j, j]]
        # value
        min_val = g_b'*d_b + (d_b'*H_b*d_b) / 2
    else
        # compute alphas
        best_block = [coord_1, eval_order[2]]
        i, j = best_block
        alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H, g, kernel, w_old, b_old)
        # get gradient
        g_b = g[[i, j]]
        # compute d
        d_b = [alpha_i-alpha[i], alpha_j-alpha[j]]
        # compute H
        H_b = [H[i, i] H[i, j]; H[j, i] H[j, j]]
        # value
        min_val = g_b'*d_b + (d_b'*H_b*d_b) / 2
    end

    # iterate through blocks
    for i = eval_order
        if all(coord_1 .!= i)
            # pick blocks in random order
            current_block = [coord_1, i]
            i, j = current_block
            # evaluate SMO rule
            alpha_prime_i, alpha_prime_j = smo_block(current_block, alpha, X, y, C, H, g, kernel, w_old, b_old)

            # get gradient
            g_b = g[[i, j]]
            # compute d
            d_b = [alpha_prime_i-alpha[i], alpha_prime_j-alpha[j]]
            # compute H
            H_b = [H[i, i] H[i, j]; H[j, i] H[j, j]]
            # value
            obj_val = g_b'*d_b + (d_b'*H_b*d_b) / 2

            # check if we need to update
            if min_val > obj_val
                # update alphas
                alpha_i, alpha_j = alpha_prime_i, alpha_prime_j
                # update min value found
                min_val = obj_val
                # best block
                best_block = current_block
            end
        end
    end
    # set stopping flag
    stop_flag = 1*(min_val == 0.)
    # return info
    return best_block, alpha_i, alpha_j, stop_flag
end

# Min grad over block
function approx_gsq_rule_2_b(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w_old, b_old)
    # init minimum
    n = size(y)[1]
    # compute gradient
    g = H * alpha - ones(n)
    # pick the first coordinate by largest gradient not equal to
    viable_indices = findall(((g .>= 0) .& (alpha .== C)) .| ((g .<= 0) .& (alpha .== 0.)) .| ((alpha .< C) .& (alpha .> 0.)))
    # set the values we can actually update
    viable_g = g[viable_indices]
    viable_alpha = alpha[viable_indices]
    # get max coordinates
    coord_1 = findall(viable_g .== minimum(viable_g))
    # check that there are not multiples
    if length(coord_1) >= 1
        eval_order = shuffle(collect(1:length(coord_1)))
        coord_1 = viable_indices[coord_1[eval_order[1]]]
    end
    # now reshuffle eval order
    eval_order = shuffle(collect(1:n))
    # init new alpha updates
    if all(coord_1 .!= eval_order[1])
        # compute alphas
        best_block = [coord_1, eval_order[1]]
        i, j = best_block
        alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H, g, kernel, w_old, b_old)
        # get gradient
        g_b = g[[i, j]]
        # compute d
        d_b = [alpha_i-alpha[i], alpha_j-alpha[j]]
        # compute H
        H_b = [H[i, i] H[i, j]; H[j, i] H[j, j]]
        # value
        min_val = g_b'*d_b + (d_b'*H_b*d_b) / 2
    else
        # compute alphas
        best_block = [coord_1, eval_order[2]]
        i, j = best_block
        alpha_i, alpha_j = smo_block(best_block, alpha, X, y, C, H, g, kernel, w_old, b_old)
        # get gradient
        g_b = g[[i, j]]
        # compute d
        d_b = [alpha_i-alpha[i], alpha_j-alpha[j]]
        # compute H
        H_b = [H[i, i] H[i, j]; H[j, i] H[j, j]]
        # value
        min_val = g_b'*d_b + (d_b'*H_b*d_b) / 2
    end

    # iterate through blocks
    for i = eval_order
        if all(coord_1 .!= i)
            # pick blocks in random order
            current_block = [coord_1, i]
            i, j = current_block
            # evaluate SMO rule
            alpha_prime_i, alpha_prime_j = smo_block(current_block, alpha, X, y, C, H, g, kernel, w_old, b_old)

            # get gradient
            g_b = g[[i, j]]
            # compute d
            d_b = [alpha_prime_i-alpha[i], alpha_prime_j-alpha[j]]
            # compute H
            H_b = [H[i, i] H[i, j]; H[j, i] H[j, j]]
            # value
            obj_val = g_b'*d_b + (d_b'*H_b*d_b) / 2

            # check if we need to update
            if min_val > obj_val
                # update alphas
                alpha_i, alpha_j = alpha_prime_i, alpha_prime_j
                # update min value found
                min_val = obj_val
                # best block
                best_block = current_block
            end
        end
    end
    # set stopping flag
    stop_flag = 1*(min_val == 0.)
    # return info
    return best_block, alpha_i, alpha_j, stop_flag
end

# Fit function
function fit_gsq_approx_2(X, y, X_test, y_test, kernel, C, epsilon, max_iter, print_info_)

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
        best_block, alpha_i, alpha_j, stop_flag_1 = approx_gsq_rule_2_a(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w, b)

        # Set new alpha values
        alpha[Int(best_block[1])] = alpha_i
        alpha[Int(best_block[2])] = alpha_j

        count_ += 1

        # compute best block
        best_block, alpha_i, alpha_j, stop_flag_2 = approx_gsq_rule_2_b(blocks, number_of_blocks, alpha, X, y, C, H, kernel, w, b)

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
        stop_flag = stop_flag_1*stop_flag_2
        satified, testErr, trainErr, alpha = stopping_conditions(testErr,trainErr,X,y,n,alpha,w,b,count_,max_iter,C,epsilon,stop_flag)

        # check if we should stop
        if satified
            break
        end
    end

    # compute parameters
    sv = findall((alpha .> 0) .& (alpha .< C))
    # return it all
    return trainErr, testErr, count_, sv
end
