function piecewise_linear(y,gradient,I1,I2)
# a piecewise_linear equation solver, y is the label vector, gradient is
# the gradient vector, I1 and I2 are index sets

    function solve(lowerbound,upperbound)

        leftside = 0
        rightside = 0

        for i = 1:length(y)
            if i in I1
                if y[i] == 1 && lowerbound > gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                elseif y[i] == -1 && upperbound < gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                end
            elseif i in I2
                if y[i] == 1 && upperbound < gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                elseif y[i] == -1 && lowerbound > gradient[i]
                    leftside += 1
                    rightside += y[i]*gradient[i]
                end
            else
                leftside +=1
                rightside += y[i]*gradient[i]
            end
        end

        if leftside == 0
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

    println(lammy)

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
    return d
end

piecewise_linear([1,-1],[2,3],[],[1])
