using LinearAlgebra
# f(x) = x'Ax
# minimize f(x)
# subject to x >= 0, z'x = 0

function f(A,x)
    return x'*A*x
end

function h(z,y)
    n = length(y)+1
    proj_x = vcat(y, [z[n,n]*(y'*z)])
    return f(proj_x)
end

function gradient_f(A,x)
    return x'*A
end

function gradient_h(A,z,y)
    # gradient
    g = zeros(length(y))
    n = length(y)+1
    for k = 1:length(y)
        for i = 1:length(y)
            g[k] += y[i]*(z[n]*z[i]*A[n,k]+z[k]*z[i]*A[k,n]+A[i,k]+z[n]^2*z[i]*z[k]*A[n,n])
        end
        g[k] += y[k]*(z[n]*z[k]*A[n,k]+z[k]*z[k]*A[k,n]+A[k,k]+z[n]^2*z[k]*z[k]*A[n,n])
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
            H[k,q] = z[n]*z[q]*A[n,k]+z[n]*z[q]*A[k,n]+A[q,k]+z[n]*z[q]*z[k]*A[n,n]
            if k==q
                H[k,q] += z[n]*z[k]*A[n,k]+z[n]*z[k]*A[k,n] + A[k,k] + z[n]^2*z[k]^2*A[n,n]
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
