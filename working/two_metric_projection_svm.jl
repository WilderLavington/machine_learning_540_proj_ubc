
# load the required packages
using LinearAlgebra
# load helpers for printing and KKTs
include("helper_fxns.jl")

# problem
# f(x) = x'Ax - sum(x)
# minimize f(x)
# subject to x >= 0, z'x = 0

function f(A,x)
    obj = x'*A*x
    # svm addtion
    obj = obj[1] - sum(x)
    return obj
end

function h(A,z,y)
    n = length(y)+1
    proj_x = vcat(y, [-1*z[n]*(y'*z[1:end-1])])
    return f(A,proj_x)
end

function gradient_h(A,z,y)
    # gradient
    g = zeros(length(y))
    n = length(y)+1
    for k = 1:length(y)
        for i = 1:length(y)
            g[k] += y[i]*(A[i,k]-z[n]*z[i]*(A[k,n]+A[n,k]) + A[n,n]*z[i]*z[k])
        end
        # svm addtion
        g[k] += y[k]*(A[k,k]-z[n]*z[k]*(A[k,n]+A[n,k]) + A[n,n]) - 1 + z[n]*z[k]
    end
    return g
end

function Hessian_h(A,z,y)
    n = length(y)+1
    H = zeros(n-1,n-1)
    for k =1:n-1
        for q = 1:n-1
            H[k,q] = A[q,k] - z[n]*z[q]*(A[k,n]+A[n,k])+A[n,n]*z[q]*z[k]
            if k==q
                H[k,q] += A[q,k] - z[n]*z[q]*(A[k,n]+A[n,k])+A[n,n]
            end
        end
    end
    return H
end

function two_metric_step(A,y,zeta,C,step_size)

    # compute gradient of h(y)
    g = gradient_h(A,y,zeta)

    # compute hessian
    H = Hessian_h(A,y,zeta)

    # find set of indices at boundary
    indicator = ((zeta .== 0) .& (g .> 0)) .| ((zeta .== C) .& (g .< 0))
    I_bound = findall(indicator)
    I_not = findall(.!indicator)

    # compute diagnol update matrix
    if !isempty(I_bound)
        d_1 = H[I_not, I_not]
        d_2 = abs.(Matrix(Diagonal(H[I_bound, I_bound])))
        fill = zeros((length(I_not),length(I_bound)))
        D_1 = [d_1 fill]
        D_2 = [fill' d_2]
        D = [D_1;D_2]
    else
        D = H
    end

    # compute one update step
    zeta_new = zeta - step_size * (D\g)

    # now project
    zeta_new = max.(zeros(length(zeta_new)), zeta_new)
    zeta_new = min.(C.*ones(length(zeta_new)), zeta_new)

    # return the new zeta
    return zeta_new
end

function linesearch(f, gradf, zeta, A, y)
    # required improvement
    c = 10^-4
    # step size drop at every iteration
    p = .9
    # always assume 1 since near the optima the step size converges to 1.
    t = 1.
    # amortize grad calc
    pk = -gradf(A, y, zeta)
    # check conditions (armino ?)
    while f(A, y, zeta + t*pk) > f(A, y, zeta)+c*t*gradf(A, y, zeta)'*pk
        t = p*t
    end
    # return the step size
    return t
end

# first two metric projection scheme
function two_metric_projection(X, y, X_test, y_test, kernal_func, C, epsilon, max_iter, print_info_)

    # pre-compute hessian
    A = (y * y').*(X * X')
    A_test = (y_test * y_test').*(X_test * X_test')

    # Initializations
    n, d = size(X)
    alpha = zeros(n)
    zeta = alpha[1:end-1]

    # data storage
    trainErr = zeros(Int(max_iter))
    testErr = zeros(Int(max_iter))

    # count for maxing out iterations
    count_ = 1

    # Compute model parameters
    sv = findall((alpha .> 0) .& (alpha .< C))

    w = transpose(X) * (alpha.*y)
    if length(sv) > 0
        b = transpose(w) * X[sv[1],:] - y[sv[1]]
    else
        b = 0
    end

    # perform linesearch
    step_size = linesearch(h, gradient_h, zeta, A, y)

    # take the first step
    zeta = two_metric_step(A,y,zeta,C,step_size)

    # compute the new x
    alpha = vcat(zeta, [-1*y[n]*(zeta'*y[1:end-1])])

    # Evaluation
    trainErr[count_] = 0.5*alpha'*(A)*alpha - sum(alpha)
    testErr[count_] = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]

    # print info
    if print_info_
        print_info(count_, trainErr[count_], testErr[count_])
    end

    for i = 1:max_iter

        # update stopping conditions
        count_ += 1

        # perform linesearch
        step_size = linesearch(h, gradient_h, zeta, A, y)

        # take the first step
        zeta = two_metric_step(A,y,zeta,C,step_size)

        # compute the new x
        alpha = vcat(zeta, [-1*y[n]*(zeta'*y[1:end-1])])

        # re-compute model parameters
        sv = findall((alpha .> 0) .& (alpha .< C))
        w = transpose(X) * (alpha.*y)
        if length(sv) > 0
            b = transpose(w) * X[sv[1],:] - y[sv[1]]
        else
            b = 0
        end

        # Evaluation
        trainErr[count_] = 0.5*alpha'*(A)*alpha - sum(alpha)
        testErr[count_] = sum((predict(X_test, w, b) .!= y_test))/size(y)[1]

        # print info
        if print_info_
            print_info(count_, trainErr[count_], testErr[count_])
        end

        # stopping condistions
        satified, testErr, trainErr, alpha = stopping_conditions(testErr,trainErr,X,y,n,alpha,w,b,count_,max_iter,C,epsilon,0)

        # check if we should stop
        if satified
            break
        end
    end
    println(y'*alpha)
    return trainErr, testErr, count_, alpha
end
