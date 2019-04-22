using LinearAlgebra
using Statistics

# Two-metric projection for constrained smooth convex problems, for details
# please read Optimization for Machine Learning Chapter 11
function twoMetric(l,u,g,H,x,eps)
# l is the lower bound, u is the upper bound. g is the gradient and H is the
# Hessian, x is the current value of the variable, eps defines the restricted
# variable set

# m is the dimension
m = size(x)[1]

# res is the restricted variable set
res = []
for i in 1:m
    if (x[i]<=l[i]+eps && g[i]>0) || (x[i]>=u[i]-eps && g[i]<0)
        res = [res;i]
    end
end

# Sbar is defined as in the textbook
Sbar = inv(H[res,res])

# S is defined as in the textbook, here we choose D to be identity matrix
S = zeros(m,m)+I
S[res,res] = Sbar

# Line search to compute alpha
#TODO

# xnew is the new value of x
xnew = x-alpha*S*g
return xnew = median([l xnew u],dims=2)

end
