function lambda_solver(d, a, c, lo, b)

    breakpoints = unique(append!(a - d.*b,a - d.*lo))
    breakpoints = sort(breakpoints)
    l = 1
    r = length(breakpoints)
    L = sum(b)
    R = sum(lo)

    # Step 1
    while true
        if r-l == 1

    # Step 4
            return y[l]+(y[r]-y[l])*(c-L)/(R-L)
        else
            global m
            m = floor((l+r)/2)
        end

    # Step 2
        C = sum(max.(min.((a.-y[m])./d,b),l))

    # Step 3
        if C=c
            return y[m]
        elseif C>c
            l = m
            L = C
        elseif C<c
            r = m
            R = C
        end
    end
end

function projection(d, a, c, lo, b)
# This function find the projection of x onto the contraints
# d is the diagonal of the quadratic function, a is the coefficient
# c is the sum, lo is the lowerbound and b is the upperbound
# see Helgason's paper for this algorithm
    lammy = lambda_solver(d, a, c, lo, b)
    n = length(d)
    x = zeros(n)
    for i = 1:n
        if lammy <= a[i]-d[i]*b[i]
            x[i] = b[i]
        elseif a[i]-d[i]*b[i] < lammy < a[i]-d[i]*lo[i]
            x[i] = (a[i]-lammy)/d[i]
        else
            x[i] = lo[i]
        end
    end
    return x, lammy
end

function piecewise_linear(y,gradient,I1,I2)
    n = length(y)
    b = zeros(n) .+ Inf
    lo = zeros(n) .- Inf
    for i = 1:n
        if in(I1).(i)
            lo[i] = 0
        end

        if in(I2).(i)
            b[i] = 0
        end
    end

    (d,lammy) = projection(y, -gradient, 0, lo, b)

    I1t = index[in(I1).(index) .& y*lammy .< gradient]
    I2t = index[in(I2).(index) .& y*lammy .> gradient]
    return (d, I1t, I2t)
end

# This function computes a descent direction d
function two_metric_per_ite(X, y, C, alpha, H)
    # H is the Hessian
    # Step 1
    n, d = size(X)
    gradient = H * alpha - ones(size(y))
    index = Array(1:n)
    I1 = index[alpha.==0]
    I2 = index[alpha.==C]
    (dx, I1t, I2t) = piecewise_linear(y,gradient,I1,I2)
    # Step 2
    dxplus = -(dx+gradient)
    # Step 3
    D = zeros(n,n) + I
    for i = 1:n
        for j = 1:n
            if ((i in I1t) || (i in I2t)) && ((j in I1t) || (j in I2t))
                D[i,j] = H[i,j]
            end
        end
    end
    # Step 4
    Dd = D*dx
    ynew = y
    ynew[in(I1t).(index) .| in(I2t).(index)] .= 0
    (dxt, a, b) = piecewise_linear(ynew,-Dd,I1,I2)
    dxt[in(I1t).(index) .| in(I2t).(index)] .= 0
    return -(dxt+dxplus)
end
