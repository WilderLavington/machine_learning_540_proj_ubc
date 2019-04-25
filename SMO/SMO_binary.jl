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
        return sign.(x*w .+ b)
    else
        return sign.(w'*x + b)
    end
end
# fit function
function fit(X, y, kernel, C, epsilon, max_iter)
    # Initializations
    n, d = size(X)
    alpha = zeros(n)
    count = 0
    # primary loop
    while true
        alpha_prev = alpha
        for j = 1:n
            count += 1
            # get random integer between 0, and n-1 != j
            i = resrndint(0, n, j)
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

                # Compute model parameters
                w = transpose(X) * (alpha.*y)
                b = mean(y .- transpose(w)*transpose(X))

                # E_i, and E_j (prediction error)
                E_i = predict(x_i, w, b) - y_i
                E_j = predict(x_j, w, b) - y_j

                # Set new alpha values
                alpha[j] = alpha_prime_j + (y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
            end
        end
        # Check convergence via KKT
        KKT_1 = findall(predict(X, w, b) .* y .>= 1) == findall(alpha .== 0)
        KKT_2 = findall((alpha .>= 0.0) .& (alpha .<= C)) == findall(predict(X, w, b) .* y .== 1)
        KKT_3 = findall(alpha .== C) == findall(predict(X, w, b) .* y .<= 1)
        if KKT_1 & KKT_2 & KKT_3
            break
        # stopping condistions
        elseif count >= max_iter
            print("exceeded max iterations")
            break
        end
    end
    # Compute model parameters
    w = transpose(X) * (alpha.*y)
    b = mean(y .- transpose(w)*transpose(X))
    # Get support vectors
    alpha_idx = findall(0 .<  alpha)
    support_vectors = X[alpha_idx, :]
    return support_vectors, count, w, b
end
# generate all combinates
function generate_blocks(n)
    blocks = zeros(n*(n-1),2)
    count = 0
    for i = 1:n
        for j = 1:n
            if i != j
                count += 1
                blocks[count, 1] = Int(i)
                blocks[count, 2] = Int(j)
            end
        end
    end
    return blocks
end
# min over block
function gsq_block(block, alpha, x, y, C, L)
    # evaluate expression
    g = (y * y').*(x * x') * alpha - ones(size(y))
    alpha_star = (g[Int(block[1])] - g[Int(block[2])]) / (2*L)
    if alpha_star <= min(C-alpha[Int(block[1])],alpha[Int(block[2])])
        alpha_star = min(C-alpha[Int(block[1])],alpha[Int(block[2])])
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
# min over block
function gsq_rule(blocks, alpha, X, y, C, L)
    updates = zeros(length(blocks))
    for i = 1:size(blocks)[1]
        updates[i] = gsq_block(blocks[i,:], alpha, X, y, C, L)
    end
    idx = argmin(updates)
    return blocks[idx,:]
end
# fit function
function fit_gsq(X, y, kernel, C, epsilon, max_iter)
    # define lipshizt
    L = maximum((y * y').*(X * X'))
    # generate all blocks
    blocks = generate_blocks(length(y))
    # Initializations
    n, d = size(X)
    alpha = (Matrix{Float64}(I, length(y), length(y)) - (y*y')./(y'*y))*rand(n)./C

    count = 0
    # primary loop
    while true
        # update stopping conditions
        count += 1
        alpha_prev = alpha
        # compute best block
        best_block = gsq_rule(blocks, alpha, X, y, C, L)
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

            # Compute model parameters
            w = transpose(X) * (alpha.*y)
            b = mean(y .- transpose(w)*transpose(X))

            # E_i, and E_j (prediction error)
            E_i = predict(x_i, w, b) - y_i
            E_j = predict(x_j, w, b) - y_j

            # Set new alpha values
            alpha[j] = alpha_prime_j + (y_j * (E_i - E_j))/k_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)
            alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
        end

        # Check kkt conditions
        if
            break
        # stopping condistions
        elseif count >= max_iter
            print("exceeded max iterations")
            break
        end
    end
    # Compute model parameters
    w = transpose(X) * (alpha.*y)
    b = mean(y .- transpose(w)*transpose(X))
    # Get support vectors
    alpha_idx = findall(0 .<  alpha)
    support_vectors = X[alpha_idx, :]
    return support_vectors, count, w, b
end
# import data
using RDatasets, LIBSVM

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")
# LIBSVM handles multi-class data automatically using a one-against-one strategy
labels = iris[:Species]
# First dimension of input data is features; second is instances
instances = iris[:, 1:4]

# set training data
X_train = convert(Matrix, instances[1:2:end, :])
y_train = labels[1:2:end]

# set test data
X_test = convert(Matrix, instances[2:2:end, :])
y_test =  labels[2:2:end]

""" BINARY CLASSIFICATION """

# binary labels
label_1train = findall(y_train .==  "versicolor")
label_2train = findall(y_train .==  "setosa")
label_1test = findall(y_test .==  "versicolor")
label_2test = findall(y_test .==  "setosa")
binary_labels_train = [label_1train;label_2train]
binary_labels_test = [label_1test;label_2test]

# set training data
X_train_bin = X_train[binary_labels_train, :]
y_train_bin = zeros(length(binary_labels_train))
y_train_bin[label_1train] .= 1
y_train_bin[label_2train] .= -1

# set test data
X_test_bin = X_test[binary_labels_test, :]
y_test_bin =  zeros(length(binary_labels_test))
y_test_bin[label_1test] .= 1
y_test_bin[label_2test] .= -1

# hyper parameters
max_iter = 10000
kernal_func = linear_kernal
C = 1.0
epsilon = 0.000001

# train model
support_vectors, count, w, b = fit(X_train_bin, y_train_bin, kernal_func, C, epsilon, max_iter)
support_vectors, count, w, b = fit_gsq(X_train_bin, y_train_bin, kernal_func, C, epsilon, max_iter)

# look at test error
pred = predict(X_test_bin, w, b)
print(sum((pred .!= y_test_bin)))

X_fake = rand(100,2)
X_fake[1:50,:] = X_fake[1:50,:] - 2*rand(50,2)
y_fake = ones(100)
y_fake[1:50] = -1*ones(50)

support_vectors, count, w, b = fit(X_fake, y_fake, kernal_func, C, epsilon, max_iter)
pred = predict(X_fake, w, b)
print(sum((pred .!= y_fake)))

support_vectors, count, w, b = fit_gsq(X_fake, y_fake, kernal_func, C, epsilon, max_iter)
pred = predict(X_fake, w, b)
print(sum((pred .!= y_fake)))


using Plots
scatter([X_fake[findall(pred.==1.),1], X_fake[findall(pred.==-1.),1]],[X_fake[findall(pred.==1.),2], X_fake[findall(pred.==-1.),2]],title="My Scatter Plot")
scatter([X_fake[findall(y_fake.==1.),1], X_fake[findall(y_fake.==-1.),1]],[X_fake[findall(y_fake.==1.),2], X_fake[findall(y_fake.==-1.),2]],title="My Scatter Plot")

x = 1:10; y = rand(10,1) # 2 columns means two lines
p = plot(x,y)
z = rand(10)
plot!(p,x,z)
