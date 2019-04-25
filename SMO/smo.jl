using Random
using LinearAlgebra
using Statistics

# linear kernal
function linear_kernal(x1, x2)
    return x1'*x2
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
    println("error in ur codes ~ 26")
    return 0
end

# Predict function
function predict(x, w, b)
    if length(x) > length(w)
        return sign.(x*w .- b)
    else
        return sign.(w'*x - b)
    end
end

# smo update
function smo_update(block, X, y, alpha, kernal, w_old, b_old)

    # compute blocks
    i, j = Int(block[1]), Int(block[2])

    # pick the x and ys for the update
    x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]

    # evaluate the kernal under these values
    k_ij = kernal(x_i, x_i) + kernal(x_j, x_j) - 2 * kernal(x_i, x_j)

    # if the points are orthogonal pass on
    if k_ij == 0
        # Compute model parameters
        new_w = w_old
        new_b = b_old
        return new_w, new_b, alpha
    else
        # get the current dual parameters
        alpha_j, alpha_i = alpha[j], alpha[i]

        # get H and L values
        if y_i != y_j
            (L, H) = (max(0, alpha_j - alpha_i), min(C, C - alpha_i + alpha_j))
        else
            (L, H) = (max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j))
        end

        # E_i, and E_j (prediction error)
        E_i = (transpose(w_old)*x_i - b_old) - y_i
        E_j = (transpose(w_old)*x_j - b_old) - y_j

        # find alpha 2 clipped
        alpha_prime_j = alpha_j + (y_j * (E_i - E_j))/k_ij
        alpha_prime_j = max(alpha_prime_j, L)
        alpha_prime_j = min(alpha_prime_j, H)

        # find alpha 1
        alpha_prime_i = alpha_i + y_i*y_j * (alpha_j - alpha_prime_j)

        # update alphas
        alpha[i] = alpha_prime_i
        alpha[j] = alpha_prime_j

        # update parameters (b)
        b_1 = E_i + y_i*(alpha_prime_i - alpha_i)*kernal(x_i, x_i) + y_j*(alpha_prime_j - alpha_i)*kernal(x_i, x_j) + b_old
        b_2 = E_j + y_i*(alpha_prime_i - alpha_i)*kernal(x_i, x_j) + y_j*(alpha_prime_j - alpha_i)*kernal(x_j, x_j) + b_old
        new_b = (b_1 + b_2) / 2

        # update parameters (w)
        new_w = w_old + y_i*(alpha_prime_i - alpha_i)*x_i + y_j*(alpha_prime_j - alpha_i)*x_j

        # return
        return new_w, new_b, alpha
    end
end

# check to see if kkt are satified
function satify_kkt(pred, alpha, y, C, n)
    satified = true
    elementwise = [false for i = 1:length(y)]
    for i = 1:n
        if alpha[i] == 0
            if pred[i]*y[i] >= 1 - epsilon
                continue
            else
                elementwise[i] = true
                satified = false
            end
        elseif alpha[i] == C
            if pred[i]*y[i] <= 1 + epsilon
                continue
            else
                elementwise[i] = true
                satified = false
            end
        else
            if (pred[i]*y[i] >= 1 - epsilon) & (pred[i]*y[i] <= 1 + epsilon)
                continue
            else
                elementwise[i] = true
                satified = false
            end
        end
    end
    return satified, elementwise
end

# pick second dual variable
function pick_second_dual(j, alpha, X, y, w, b)

    # get error of chosen block
    E_j = (transpose(w)*X[j,:] - b) - y[j]

    # predict E for all other examples
    E = (X*w .- b) - y

    # remove the current E
    deleteat!(E, j)

    # pick the other block following hueristic
    if E_j > 0
        return argmin(E)
    else
        return argmax(E)
    end
end

# compute parameters
function parameters(alpha,X,y)
    # find a support vector
    sv = findall((alpha .> 0) .& (alpha .< C))
    # Compute model parameters
    w = transpose(X) * (alpha.*y)
    if length(sv) > 0
       b = transpose(w) * X[sv[1],:] - y[sv[1]]
    else
       b = 0
    end
    return w, b
end

# Fit function
function fit_smo(X, y, kernel, C, epsilon, max_iter)

    # Initializations
    n, d = size(X)
    alpha = zeros(n)

    # inner - loop cut off
    loop_cut = 2*n

    # initialize parameters
    w, b = parameters(alpha,X,y)

    # primary loop
    count = 0
    while true

        # determine which blocks can be optimized
        pred = X*w .- b
        _, is_eligable = satify_kkt(pred, alpha, y, C, n)

        # now determine the current support vectors not on boundaries
        unbound = findall((alpha .> 0) .& (alpha .< C) .& is_eligable)
        ub = length(unbound)

        # iterate through these until they are all consistant with the kkt
        inner_count = 0
        while ub > 0

            # iterate through all unbound examples + random sub-block
            for  j = 1:ub

                # pick second according to hueristics
                i = pick_second_dual(j, alpha, X, y, w, b)

                # define block for update
                block = [i, unbound[j]]

                # update stopping conditions
                count += 1

                # perform step on block
                w, b, alpha = smo_update(block, X, y, alpha, kernel, w, b)
            end

            # determine which blocks can be optimized
            pred = X*w .- b
            _, is_eligable = satify_kkt(pred, alpha, y, C, n)

            # now determine the current support vectors not on boundaries
            unbound = findall((alpha .> 0) .& (alpha .< C) .& is_eligable)
            ub = length(unbound)

            # stopping conditions
            inner_count += 1
            if inner_count >= loop_cut
                break
            end

        end

        # determine which blocks can be optimized
        pred = X*w .- b
        _, is_eligable = satify_kkt(pred, alpha, y, C, n)
        eligable = findall(is_eligable)
        e = length(eligable)

        # now just iterate over all values that dont satisfy kkt
        for  j = 1:e
            # pick second according to hueristics
            i = pick_second_dual(eligable[j], alpha, X, y)

            # define block for update
            block = [i, eligable[j]]

            # update stopping conditions
            count += 1

            # perform step on block
            w, b, alpha = smo_update(block, X, y, alpha, kernel, w, b)
        end

        # Check convergence via KKT
        pred = X*w .- b
        satified, _ = satify_kkt(pred, alpha, y, C, n)

        # stopping condistions for bound examples
        if satified
            println("satified kkt")
            break
        elseif count >= max_iter
            println("exceeded max iterations")
            break
        end

    end

    # Get support vectors
    alpha_idx = findall((alpha .> 0) .& (alpha .< C))
    support_vectors = X[alpha_idx, :]

    # return
    return support_vectors, count, w, b
end


X_fake = rand(20,2)
X_fake[1:10,:] = X_fake[1:10,:] - 2*rand(10,2)
y_fake = ones(20)
y_fake[1:10] = -1*ones(10)

# hyper parameters
max_iter = 2e5
kernal_func = linear_kernal
C = 1.0
epsilon = 0.01

support_vectors, count, w, b = fit_smo(X_fake, y_fake, kernal_func, C, epsilon, max_iter)
pred = predict(X_fake, w, b)
print(sum((pred .!= y_fake)))
