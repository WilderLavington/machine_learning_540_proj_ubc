using Random
using LinearAlgebra
using Statistics
using Printf

# Min over block
function smo_block(block, alpha, X, y, C, H, g, kernel, w_old, b_old)
    # compute blocks
    i, j = Int(block[1]), Int(block[2])
    # pick the x and ys for the update
    x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
    # evaluate the kernal under these values
    k_ij = H[i,i] + H[j,j] - 2 * H[i,j]
    # if the points are orthogonal pass on
    if k_ij == 0
        # get the current dual parameters
        alpha_j, alpha_i = alpha[j], alpha[i]
        # Compute model parameters
        alpha_prime_i = alpha_i
        alpha_prime_j = alpha_j
        # this is a degenerate point
        return Inf, alpha_prime_i, alpha_prime_j
    else
        # get the current dual parameters
        alpha_j, alpha_i = alpha[j], alpha[i]
        # get H and L values
        if y_i != y_j
            (L, U) = (max(0, alpha_j - alpha_i), min(C, C - alpha_i + alpha_j))
        else
            (L, U) = (max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j))
        end
        # E_i, and E_j (prediction error)
        E_i = (transpose(w_old)*x_i - b_old) - y_i
        E_j = (transpose(w_old)*x_j - b_old) - y_j
        # find alpha 2 clipped
        alpha_prime_j = alpha_j + (y_j * (E_i - E_j))/k_ij
        alpha_prime_j = max(alpha_prime_j, L)
        alpha_prime_j = min(alpha_prime_j, U)
        # find alpha 1
        alpha_prime_i = alpha_i + y_i*y_j * (alpha_j - alpha_prime_j)
        # return
        return alpha_prime_i, alpha_prime_j
    end
end

# joeys version
