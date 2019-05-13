using Random
using LinearAlgebra
using Statistics
using Printf
# linear kernal
function linear_kernal(x1, x2)
    return x1'*x2
end
function predict(x, w, b)
    if length(x) > length(w)
        return sign.(x*w .- b)
    else
        return sign.(w'*x .- b)
    end
end
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
function randomfit(X, y, X_test, y_test, kernel, C, epsilon, max_iter)
    # Initializations
    trainErr = zeros(Int(max_iter))
    testErr = zeros(Int(max_iter))
    n, d = size(X)
    alpha = zeros(n)
    count_ = 1

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
        count_ += 1
        # get random integer between 0, and n-1 != j
        j = rand(1:n)
        i = resrndint(0, n-1, j)
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