# objective functon
function objective(y, x, alpha)
	f = alpha' * (y * y').*(x * x') * alpha - alpha*ones(size(y))
	g = (y * y').*(x * x') * alpha - ones(size(y))
	H = (y * y').*(x * x')
	return (f, g, H)
end

# This function compute a basis for the null space under block constraints
function computeBasis(x,n)
	# x specifies a block, n is the dimensionality
	m = size(x)[1]
	# B is the basis we compute
	B = []
	for i in 2:m
	    b = zeros(n,)
	    b[x[1]] = 1
	    b[x[i]] = -1
	    B = [B b]
	end
	return B
end


function fit(X, y, C, epsilon, max_iter, Numblocks)
    # Initializations
    n, d = size(X)
    alpha = randn(n) ./ (C + 1)
    count = 0

    # set blocks
    blocks = reshape(randperm(length(y)), (Numblocks, Int(length(y) / Numblocks)))

    # initial stepsize
    step_size = 1

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

		# get update info
		(f, g, H) = objective(y, x, alpha)

		# get basis
		B = get_basis(alpha_b, n_b)

		# take gradient and use nullspace projection
		alpha_bNew = (B*inv(B'*B)*B')*(x_bNew - step_size * g)

		# return all alpha that no longer satify constraints
		alpha_lower = findall(0 .>  alpha_bNew)
		alpha_higher = findall(C .<  alpha_bNew)
		alpha_other = findall((alpha_bNew .> 0) .& (alpha_bNew.< C))

		# take the real projection step
		alpha_b[alpha_lower] = (B*inv(B'*B)*B')*(zeros(length(alpha_higher)))
		alpha_b[alpha_higher] = (B*inv(B'*B)*B')*(C*ones(length(alpha_higher)))
		alpha_b[alpha_other] = alpha_bNew[alpha_other]

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
