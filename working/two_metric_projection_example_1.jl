using LinearAlgebra
# f(x) = x'Ax
# minimiz f(x)
# subject to x >= 0, sum(x) = 1
# solution is x = [0,0,1]

function f(A,x)
    return x'*A*x
end

function h(y)
    proj_x = vcat(y, [1-sum(y)])
    return f(proj_x)
end

function gradient_f(A,x)
    return x'*A
end

function gradient_h(A,y)
    # gradient
    g = zeros(length(y))
    n = length(y)+1
    for k = 1:length(y)
        g[k] = y'*((A[n,n]-A[n,k]).+A[1:end-1,k]-A[n,1:end-1])
        g[k] += y[k]*(A[k,k]+A[n,n]-A[n,k]-A[k,n])
        g[k] += A[n,k] + A[k,n] - 2*A[n,n]
    end
    return g
end

function Hessian_x(A)
    return A
end

function Hessian_h(A,y)
    n = length(y)+1
    H = zeros(n-1,n-1)
    for k =1:n-1
        for q = 1:n-1
            H[k,q] = y[q]*(A[q,k]+A[n,n]-A[n,k]-A[q,n])
            if k==q
                H[k,q] += (A[k,k]+A[n,n]-A[n,k]-A[k,n])
            end
        end
    end
    return H
end

# first two metric projection scheme
function tmp_example_problem()
    # set function matrix
    A = [0.5 0.0 0;
         0.0 0.2 0;
         0 0 0.5]
    # starting guess for y
    x = [0.5, 0.2, 0.3]
    y = x[1:end-1]
    # set step size
    alpha = 0.1
    # compute gradient of h(y)
    g = gradient_h(A,y)
    # compute hessian
    H = Hessian_h(A,y)
    # find set of indices at boundary
    I_bound = findall((y .== 0) .& (g .> 0))
    I_not = findall(.!((y .== 0) .& (g .> 0)))
    # compute diagnol update matrix
    if !isempty(I_bound)
        d_1 = H[I_not, I_not]
        d_2 = Matrix(Diagonal(H[I_bound, I_bound]))
        fill = zeros((length(I_not),length(I_bound)))
        D_1 = [d_1 fill]
        D_2 = [fill' d_2]
        D = [D_1;D_2]
    else
        D = H
    end
    # compute one update step
    y_new = y - alpha * (D\g)
    # now iterate through y and set any that are less then 0 to zero
    y_new = max.(zeros(length(y_new)), y_new)
    # compute the new x
    x_new = [y_new[1] y_new[2] (1-y_new[1]-y_new[2])]'
    # now iterate
    iter = 15
    for i = 1:iter
        # compute gradient of h(y)
        g = gradient_h(A,y_new)
        # compute hessian
        H = Hessian_h(A,y_new)
        # find set of indices at boundary
        I_bound = findall((y_new .== 0) .& (g .> 0))
        I_not = findall(.!((y_new .== 0) .& (g .> 0)))
        # compute diagnol update matrix
        if !isempty(I_bound)
            d_1 = H[I_not, I_not]
            d_2 = Matrix(Diagonal(H[I_bound, I_bound]))
            fill = zeros((length(I_not),length(I_bound)))
            D_1 = [d_1 fill]
            D_2 = [fill' d_2]
            D = [D_1;D_2]
        else
            D = H
        end
        # compute one update step
        y_new = y_new - alpha * (D\g)
        # now iterate through y and set any that are less then 0 to zero

        y_new = max.(zeros(length(y_new)), y_new)
        # now project y onto null
        # update x
        x_new = [y_new[1] y_new[2] (1-y_new[1]-y_new[2])]'

        println("=====================")
        println(g)
        println(x_new)
        println(x)
        println(f(A,x))
        println(f(A,x_new)[1])
    end

    return x_new
end
