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
    print("error in ur codes ~ 26")
    return 0
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
# Min over block
function gsq_block(block, alpha, x, y, C, L)
    # evaluate expression
    g = (y * y').*(x * x') * alpha - ones(size(y))
    alpha_star = (g[Int(block[1])] - g[Int(block[2])]) / (2*L)
    if alpha_star <= min(C-alpha[Int(block[1])], alpha[Int(block[2])])
        alpha_star = min(C-alpha[Int(block[1])], alpha[Int(block[2])])
    elseif alpha_star >= max(alpha[Int(block[1])] - C, -alpha[Int(block[2])])
        alpha_star = max(alpha[Int(block[1])] - C, -alpha[Int(block[2])])
    end
    # set the update to
    alpha_bstar = alpha_star .* [1, -1]
    # get minimum
    alpha_b = zeros(size(alpha))
    alpha_b[[Int(block[1]), Int(block[2])]] = alpha[[Int(block[1]), Int(block[2])]]
    min_val = g'*alpha_b + L*(alpha_b'*alpha_b)/2
    return min_val
end
# Min over block
function gsq_rule(blocks, alpha, X, y, C, L)
    updates = zeros(size(blocks)[1])
    for i = 1:size(blocks)[1]
        updates[i] = gsq_block(blocks[i,:], alpha, X, y, C, L)
    end
    min = minimum(updates)
    val = findall(updates .== min)
    idx = val[rand(1:length(val))]
    return blocks[idx,:], idx
end
# Fit function
function fit_gsq(X, y, kernel, C, epsilon, max_iter)

    # define lipshizt
    L = opnorm((y * y').*(X * X'), Inf)

    # generate all blocks
    blocks = generate_blocks(length(y))
    print(blocks)
    # Initializations
    n, d = size(X)
    alpha = zeros(n)
    alpha[1] = C/2
    alpha[2] = -C/2
    count = 0
    current_block = []

    # primary loop
    while true
        # update stopping conditions
        count += 1
        # compute best block
        best_block, idx = gsq_rule(blocks, alpha, X, y, C, L)
        # compute blocks
        i, j = Int(best_block[1]), Int(best_block[2])
        # add back the block before this
        # if length(current_block) > 0
        #     blocks = [blocks; reshape(current_block, 1, length(current_block))]
        # end
        if length(blocks) < 0
            break
        end
        print([i, j])
        # remove the block thats been updated
        blocks = blocks[setdiff(1:end, idx), :];
        current_block = [Int(best_block[1]), Int(best_block[2])]
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

X_fake = rand(10,2)
X_fake[1:5,:] = X_fake[1:5,:] - 2*rand(5,2)
y_fake = ones(10)
y_fake[1:5] = -1*ones(5)

# hyper parameters
max_iter = 1e5
kernal_func = linear_kernal
C = 1.0
epsilon = 0.01

support_vectors, count, w, b = fit_gsq(X_fake, y_fake, kernal_func, C, epsilon, max_iter)
pred = predict(X_fake, w, b)
print(sum((pred .!= y_fake)))
