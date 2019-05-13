using Random
using LinearAlgebra
using Statistics
using Printf
# linear kernal
function linear_kernal(x1, x2)
    return x1'*x2
end
# Predict function
function predict(x, w, b)
    if length(x) > length(w)
        return sign.(x*w .- b)
    else
        return sign.(w'*x .- b)
    end
end
# Generate all combinates
function generate_blocks(n)
    blocks = zeros(Int(n*(n-1)/2),2)
    count = 0
    for i = 1:n
        for j = i:n
            if i != j
                count += 1
                blocks[count, 1] = Int(i)
                blocks[count, 2] = Int(j)
            end
        end
    end
    return blocks
end
function mymedian(num1, num2, num3)
    if num2 >= num3
        return num3
    elseif num1 >= num2
        return num1
    else
        return num2
    end
end
# Min over block
function gsq_block_diagApprx(block, alpha, X, y, C, L_val, kernel, w_old, b_old)
    # compute blocks
    i, j = Int(block[2]), Int(block[1])
    # get the current dual parameters
    alpha_j, alpha_i = alpha[j], alpha[i]
    # set g
    g = (y * y').*(X * X') * alpha - ones(size(y))
    g_b = g[[i, j]]
    # get H and L values
    (L, middle, H) = (max(- alpha_i, alpha_j - C ), -1*(g_b[1] - g_b[2])/(2*L_val) ,min(C- alpha_i, alpha_j))
    # set d
    d_b = [mymedian(L, middle, H), -1*mymedian(L, middle, H)]
    # value
    min_val = g_b'*d_b + L_val^2*(d_b'd_b)/2
    # return
    return min_val
end
# Min over block
function gsq_rule_diagApprx(blocks, alpha, X, y, C, kernel, w_old, b_old)
    updates = zeros(size(blocks)[1])
    # use the largest eigen value for picking
    H = (y * y').*(X * X')
    val = eigvals(H)
    approx = Diagonal(val)
    for i = 1:size(blocks)[1]
        updates[i] = gsq_block_diagApprx(blocks[i,:], alpha, X, y, C, approx, kernel, w_old, b_old)
    end
    min = minimum(updates)
    val = findall(updates .== min)
    idx = val[rand(1:length(val))]
    return blocks[idx,:], idx
end
# Fit function
function fit_gsq_diagApprx(X, y, X_test, y_test, kernel, C, epsilon, max_iter)

    # generate all blocks
    blocks = generate_blocks(length(y))

    # Initializations
    n, d = size(X)
    alpha = zeros(n)
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

    #Evaluation
    @printf("Iteration: %d\n",count_)
    # trainPred = predict(X, w, b)
    # trainErrRate = sum((trainPred .!= y))/size(y)[1]
    trainErr[count_] = 0.5*alpha'*((y*y').*(X*X'))*alpha - sum(alpha)
    @printf("Training error: %.3f\n", trainErr[count_])
    # testPred = predict(X_test, w, b)
    # testErrRate = sum((testPred .!= y_test))/size(y)[1]
    testErr[count_] = 0.5*alpha'*((y_test*y_test').*(X_test*X_test'))*alpha - sum(alpha)
    @printf("Testing error: %.3f\n", testErr[count_])
    alpha_prev = alpha

    # primary loop
    while true
        # update stopping conditions
        count_ += 1
        # compute best block
        best_block, idx = gsq_rule_diagApprx(blocks, alpha, X, y, C, kernel, w, b)

        # compute blocks
        i, j = Int(best_block[1]), Int(best_block[2])
        # pick the x and ys for the update
        x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
        # evaluate the kernal under these values
        k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
        # if the points are orthogonal pass on
        if k_ij == 0
            continue
        else
            # get the current dual parameters
            alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
            # get H and L values
            if y_i != y_j
                (L, H) = (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
            else
                (L, H) = (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
            end

            # find a support vector
            sv = findall((alpha .> 0) .& (alpha .< C))

            # Compute model parameters
            w = transpose(X) * (alpha.*y)
            if length(sv) > 0
                b = transpose(w) * X[sv[1],:] - y[sv[1]]
            else
                b = 0
            end

            # E_i, and E_j (prediction error)
            E_i = (transpose(w)*x_i - b) - y_i
            E_j = (transpose(w)*x_j - b) - y_j

            # Set new alpha values
            alpha[j] = alpha_prime_j + (y_j * (E_i - E_j))/k_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)
            alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
        end

        #Evaluation
        @printf("Iteration: %d\n",count_)
        # trainPred = predict(X, w, b)
        # trainErrRate = sum((trainPred .!= y))/size(y)[1]
        trainErr[count_] = 0.5*alpha'*((y*y').*(X*X'))*alpha - sum(alpha)
        @printf("Training error: %.3f\n", trainErr[count_])
        # testPred = predict(X_test, w, b)
        # testErrRate = sum((testPred .!= y_test))/size(y)[1]
        testErr[count_] = 0.5*alpha'*((y_test*y_test').*(X_test*X_test'))*alpha - sum(alpha)
        @printf("Testing error: %.3f\n", testErr[count_])
        alpha_prev = alpha

        # Check convergence via KKT
        satified = true
        pred = X*w .- b
        for i = 1:n
            if alpha[i] == 0
                if pred[i]*y[i] >= 1 - epsilon
                    continue
                else
                    satified = false
                end
            elseif alpha[i] == C
                if pred[i]*y[i] <= 1 + epsilon
                    continue
                else
                    satified = false
                end
            else
                if (pred[i]*y[i] >= 1 - epsilon) & (pred[i]*y[i] <= 1 + epsilon)
                    continue
                else
                    satified = false
                end
            end
        end
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
