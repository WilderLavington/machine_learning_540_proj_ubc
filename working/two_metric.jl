function piecewise_linear(y,gradient,I1,I2)
# a piecewise_linear equation solver, y is the label vector, gradient is
# the gradient vector, I1 and I2 are index sets

    function solve(lowerbound,upperbound)

        leftside = 0
        rightside = 0

        for i = 1:length(y)
            if i in I1
                if y[i] == 1 && lowerbound >= gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                elseif y[i] == -1 && upperbound =< -gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                end
            elseif i in I2
                if y[i] == 1 && upperbound =< gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                elseif y[i] == -1 && lowerbound >= -gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                end
            else
                leftside +=1
                rightside += y[i]*gradient[i]
            end
        end

        if leftside == 0
            return (true, upperbound)
        elseif rightside/leftside<lowerbound || rightside/leftside>upperbound
            return (false, 0)
        else
            return (true, rightside/leftside)
        end
    end

    index = Array(1:length(y))
    breakpoints = (y.*gradient)[in(I1).(index) .| in(I2).(index)]
    order = sort(breakpoints)
    if length(order) == 0
        order = [-Inf Inf]
    else
        order = [-Inf order Inf]
    end

    for i in 1:length(order)-1
        global lammy
        (valid, lammy) = solve(order[i],order[i+1])
        if valid == true
            break
        end
    end

    d = zeros(length(y))
    for i = 1:length(y)
        val = y[i]*lammy-gradient[i]
        if i in I1
            val > 0 ? d[i] = val : d[i] = 0
        elseif i in I2
            val < 0 ? d[i] = val : d[i] = 0
        else
            d[i]=val
        end
    end
    I1t = index[in(I1).(index) .& y*lammy.<gradient]
    I2t = index[in(I2).(index) .& y*lammy.>gradient]
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

function projection(y, C, x)
# This function find the projection of x onto the contraints
    function solve(lowerbound,upperbound)

        leftside = 0
        rightside = 0

        for i = 1:length(y)
            if y[i] == 1 && (lowerbound >= -x[i] && upperbound <= -x[i]+C)
                leftside += 1
                rightside += -x[i]
            else if y[i] == 1 && (lowerbound > -x[i]+C)
                rightside -= C
            elseif y[i] == -1 && (lowerbound >= x[i]-C && upperbound <= x[i])
                leftside += 1
                rightside += x[i]
            elseif y[i] == -1 && (lowerbound > x[i])
                rightside += C
            end
        end

        if leftside == 0
            return (true, upperbound)
        elseif rightside/leftside<lowerbound || rightside/leftside>upperbound
            return (false, 0)
        else
            return (true, rightside/leftside)
        end
    end

    index = Array(1:length(y))
    breakpoints = append!(y.*-x,y.*(C.-x))
    order = sort(breakpoints)
    if length(order) == 0
        order = [-Inf Inf]
    else
        order = [-Inf order Inf]
    end

    for i in 1:length(order)-1
        global lammy
        (valid, lammy) = solve(order[i],order[i+1])
        if valid == true
            break
        end
    end

    d = zeros(length(y))
    for i = 1:length(y)
        val = y[i]*lammy+x[i]
        if val < 0
            d[i] = 0
        elseif val > C
            d[i] = C
        else
            d[i]=val
        end
    end
    return d
end
