# objective functon
function objective(y, x, alpha)
	f = alpha' * ((y * y').*(x * x')) * alpha - alpha'*ones(size(y))
	g = (y * y').*(x * x') * alpha - ones(size(y))
	H = (y * y').*(x * x')
	return (f, g, H)
end

# This function compute a basis for the null space under block constraints
function computeBasis(x,n)
	# x specifies a block, n is the dimensionality
	m = size(x)[1]
	# B is the basis we compute
	B = zeros(m-1,n)
	for i in 1:m-1
	    b = zeros(n)
	    b[x[1]] = 1
	    b[x[i]] = -1
	    B[i,:] = b
	end
	return B
end

# Predict function
function predict(x, w, b)
    if length(x) > length(w)
        return sign.(x*w .+ b)
    else
        return sign.(w'*x + b)
    end
end

# projection step
function proj_step(B, C, alpha_b, step_size, g)
	# first projected step
	alpha_bNew = (B'*inv(B*B')*B)*(alpha_b - step_size * g)
	# return all alpha that no longer satify constraints
	alpha_lower = findall(0 .>  alpha_bNew)
	alpha_higher = findall(C .<  alpha_bNew)
	alpha_other = findall((alpha_bNew .> 0) .& (alpha_bNew .< C))
	# set all these values
	alpha_bNew[alpha_other] = (alpha_b - step_size * g)[alpha_other]
	alpha_bNew[alpha_lower] = zeros(length(alpha_lower))
	alpha_bNew[alpha_higher] = C*ones(length(alpha_higher))
	# take the real projection step with clipped values
	alpha_bNew = (B'*inv(B*B')*B)*(alpha_bNew)
	# return
	return alpha_bNew
end

function fit(X, y, C, epsilon, max_iter, Numblocks)
    # Initializations
    n, d = size(X)
    alpha = randn(n) ./ (C + 1)
    count = 0

    # set blocks
    blocks = reshape(randperm(length(y)), (Numblocks, Int(length(y) / Numblocks)))

    # initial stepsize + sufficient decrease parameter
    step_size = 1e-2
	gamma = 1e-4

    # primary loop
    while true

        # update counter
        count += 1
        alpha_prev = alpha

        # pick block at random
        block = blocks[rand(1:Numblocks),:]

        # set block info
		alpha_b = zeros(size(alpha))
        alpha_b[block] = alpha[block]
        n_b = length(block)

		# get objective info
		(f_b, g_b, H_b) = objective(y, X, alpha_b)

		# get basis
		B = computeBasis(block, n)

		# Try out the current step-size
		alpha_bNew = proj_step(B, C, alpha_b, step_size, g_b)
		(f_bNew, g_bNew, H_bNew) = objective(y, X, alpha_bNew)

		# Decrease the step-size if we increased the function
		gg = dot(g_b,g_b)
		while f_b > f_bNew - gamma*step_size*gg
			# Fit a degree-2 polynomial to set step-size
			step_size = step_size^2*gg/(2(f_bNew - f_b + step_size*gg))
			# Try out the smaller step-size
			alpha_bNew = proj_step(B, C, alpha_b, step_size, g_b)
			(f_bNew, g_b, H_b) = objective(y, X, alpha_bNew)
			print(f_bNew, f_b)
		end

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
using Random
using LinearAlgebra
using Statistics

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
C = 1.0
epsilon = 0.001
Numblocks = 5

# train model
support_vectors, count, w, b = fit(X_train_bin, y_train_bin, C, epsilon, max_iter, Numblocks)

# look at test error
pred = predict(X_test_bin, w, b)
print(sum((pred .!= y_test_bin)))
