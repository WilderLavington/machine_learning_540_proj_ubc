using LinearAlgebra

function lambda_solver(a, c, lo, b)

    breakpoints = unique(append!(a - b,a - lo))
    breakpoints = sort(breakpoints)
    y = breakpoints
    l = 1
    r = Int(length(breakpoints))
    L = sum(b)
    R = sum(lo)

    # Step 1
    while true
        if r-l == 1

    # Step 4
            return y[l]+(y[r]-y[l])*(c-L)/(R-L)
        else
            global m
            m = Int(floor((l+r)/2))
        end

    # Step 2
        C = sum(max.(min.(a.-y[m],b),lo))

    # Step 3
        if C == c
            return y[m]
        elseif C > c
            l = m
            L = C
        elseif C < c
            r = m
            R = C
        end
    end
end

function projection(y, a, c, lo, b)
# This function find the projection of x onto the contraints
# y is the coefficient of the linear constraint, a is the projecting point
# c is the sum, lo is the lowerbound and b is the upperbound
# see Helgason's paper for this algorithm
    n = length(y)
    index = Array(1:n)
    minus = index[y.==-1]
    a[minus] = -a[minus]
    temp = lo
    lo[minus] = -b[minus]
    b[minus] = -temp[minus]

    lammy = lambda_solver(a, c, lo, b)

    x = zeros(n)
    for i = 1:n
        if lammy <= a[i]-b[i]
            x[i] = b[i]
        elseif a[i]-b[i] < lammy < a[i]-lo[i]
            x[i] = a[i]-lammy
        else
            x[i] = lo[i]
        end
    end
    x[minus] = -x[minus]
    return x, lammy
end

function piecewise_linear(y,gradient,I1,I2,I1til,I2til)
    n = length(y)
    index = Array(1:n)
    nor = norm(gradient)
    b = zeros(n) .+ nor
    lo = zeros(n) .- nor
    for i = 1:n
        if i in I1
            lo[i] = 0
        end

        if i in I2
            b[i] = 0
        end

        if i in I1til || i in I2til
            lo[i] = 0
            b[i] = 0
        end
    end

    (d,lammy) = projection(y, gradient, 0, lo, b)

    lammy = -lammy

    I1t = index[(in(I1).(index)) .& (y*lammy .< gradient)]
    I2t = index[(in(I2).(index)) .& (y*lammy .> gradient)]
    return (d, I1t, I2t)
end

# This function computes a descent direction d
function two_metric_per_ite(y, gradient, C, alpha, H)
    # H is the Hessian
    # Step 1
    n = length(y)
    gradient = H * alpha - ones(size(y))
    index = Array(1:n)
    I1 = index[alpha.==0]
    I2 = index[alpha.==C]
    (dx, I1t, I2t) = piecewise_linear(y,-gradient,I1,I2,[],[])
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
    (dxt, ph1, ph2) = piecewise_linear(y,Dd,I1,I2,I1t,I2t)
    return -(dxt+dxplus)
end
