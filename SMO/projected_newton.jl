using Random
using LinearAlgebra
using Statistics
# linear kernal
function linear_kernal(x1, x2)
    return x1'*x2
end
# hessian
function hessian(y, x)
    return (y * y').*(x * x')
end
# gradient
function gradient(y, x, alpha)
    return (y * y').*(x * x') * alpha - ones(size(y))
end
# projection
function proj(y, n, alpha)
    return (Matrix{Float64}(I, n, n) - (y*y')./(y'*y))*alpha
end
# projected newton
function proj_newton(y, x)
    # initialize
    n = length(y)
    # initialize data
    alpha = rand(size(y))
    # project alpha onto nullspace
    alpha = proj(y, n, alpha)
    # compute hessian
    H = hessian(y, x)
    # training
    for epoch = 1:epochs
        # compute gradient
        grad = gradient(y, x, alpha)
        # set up solve for update
        update = H / grad
        # newton step
        alpha = alpha - step*update
        # project update onto the nullspace
        alpha = proj(y, n, alpha)
    end
    # return values
    return alpha
end

# block projected newton
function CBCD(y, x, Numblocks)
    # set blocks
    blocks = reshape(randperm(length(y)), length(y) / Numblocks)
    # initialize
    n = length(y)
    # initialize data
    alpha = zeros(size(y))
    # training
    for epoch = 1:epochs

        # pick block at random
        block = blocks[rand(1:Numblocks),:]

        # set block info
        y_b = y[block]
        x_b = x[block]
        alpha_b = alpha[block]

        # compute hessian
        H_b = hessian(y_b, x_b)
        # compute gradient
        grad_b = gradient(y_b, x_b, alpha_b)
        # set up solve for update
        update = H_b / grad_b
        # newton step
        alpha[block] = alpha[block] - step*update

        # project update onto the nullspace
        alpha[block] = proj(y_b, n, alpha[block])

    end
    # return values
    return alpha
end


function fit(X, y, C, epsilon, max_iter, Numblocks)
    # Initializations
    n, d = size(X)
    alpha = randn(n) ./ (C + 1)
    count = 0

    # set blocks
    blocks = reshape(randperm(length(y)), (Numblocks, Int(length(y) / Numblocks)))

    # initialize stepsize
    step_size = 0.01

    # primary loop
    while true

        # update counter
        count += 1
        alpha_prev = alpha

        # pick block at random
        block = blocks[rand(1:Numblocks),:]

        # set block info
        y_b = y[block]
        x_b = X[block,:]
        alpha_b = alpha[block]
        n_b = length(block)

        # compute hessian
        H_b = hessian(y_b, x_b)

        # compute gradient
        grad_b = gradient(y_b, x_b, alpha_b)

        # set up solve for update
        update = H_b \ grad_b

        # Try out the current step-size
		alphaNew = alpha - step_size*update
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gg = dot(g,g)
		while fNew > f - gamma*alpha*gg

			if verbose
				@printf("Backtracking, step size = %f, fNew = %f\n",alpha,fNew)
			end

			if isfinitereal(fNew)
				# Fit a degree-2 polynomial to set step-size
				alpha = alpha^2*gg/(2(fNew - f + alpha*gg))
			else
				alpha /= 2
			end

			# Try out the smaller step-size
			wNew = w - alpha*g
			(fNew,gNew) = funObj(wNew)
		end

        # newton step
        alpha[block] = alpha_b - step.*update
        # project update onto the nullspace
        alpha[block] = proj(y_b, n_b, alpha_b)
        # Check convergence
        diff = norm(alpha - alpha_prev)
        if diff < epsilon
            print("broke threshold")
            break
        # stopping condistions
        elseif count >= max_iter
            print("exceeded max iterations")
            break
        end
    end
    # weights and biases
    w = transpose(X) * (alpha.*y)
    b = mean(y .- transpose(w)*transpose(X))
    # Get support vectors
    alpha_idx = findall(0 .<  alpha)
    support_vectors = X[alpha_idx, :]
    # return
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
epsilon = 0.001
Numblocks = 5

# train model
support_vectors, count, w, b = fit(X_train_bin, y_train_bin, C, epsilon, max_iter, Numblocks)

# look at test error
pred = predict(X_test_bin, w, b)
print(sum((pred != y_test_bin)))
