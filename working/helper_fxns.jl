# linear kernal
function linear_kernal(x1, x2)
    return x1'*x2
end
# quadratic kernal
function quadratic_kernal(x1, x2)
    return (1+x1'*x2)^2
end
# quadratic kernal
function cubic_kernal(x1, x2)
    return (1+x1'*x2)^3
end
# quartic kernal
function quartic_kernal(x1, x2)
    return (1+x1'*x2)^4
end
# # rbf kernal
# function rbf_kernal(x1, x2, sigma = 0.1)
#     return exp.(-1*(x1-*x2)'(x1-*x2) / sigma^2)
# end
# get diagnol matrix
function diagnolize(matrix)
    n, m = size(matrix)
    if n != m
        return None
    else
        diag_mat = zeros(n,m)
        for i = 1:n
            diag_mat[i,i] = matrix[i,i]
        end
        return diag_mat
    end
end
# predict function
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
# something to print stuff
function print_info(count_, trainErr, testErrRate)
    @printf("Iteration: %d\n",count_)
    @printf("Objective Function: %.3f\n", trainErr)
    @printf("Testing error: %.3f\n", testErrRate)
end
# Generate all combinates
function generate_blocks_j(n,j)
    blocks = zeros(n-1,2)
    count = 0
    for i = 1:n
        if i != j
            count += 1
            blocks[count, 1] = Int(i)
            blocks[count, 2] = Int(j)
        end
    end
    return blocks
end
function generate_blocks_i(n,i)
    blocks = zeros(n-1,2)
    count = 0
    for j = 1:n
        if i != j
            count += 1
            blocks[count, 1] = Int(i)
            blocks[count, 2] = Int(j)
        end
    end
    return blocks
end
# kkt conditions
function KKT_conditions(X,y,n,alpha,w,b)
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
    return satified
end
