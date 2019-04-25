using Random
using LinearAlgebra
using Statistics
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
function randomfit(X, y, kernel, C, epsilon, max_iter)
    # Initializations
    n, d = size(X)
    alpha = zeros(n)
    count = 0
    # primary loop
    while true
        alpha_prev = alpha
        count += 1
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
        elseif count >= max_iter
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
    return support_vectors, count, w, b
end

X_fake = rand(20,2)
X_fake[1:10,:] = X_fake[1:10,:] - 2*rand(10,2)
y_fake = ones(20)
y_fake[1:10] = -1*ones(10)

# hyper parameters
max_iter = 1e6
kernal_func = linear_kernal
C = 1.0
epsilon = 0.01

support_vectors, count, w, b = randomfit(X_fake, y_fake, kernal_func, C, epsilon, max_iter)
pred = predict(X_fake, w, b)
print(sum((pred .!= y_fake)))
