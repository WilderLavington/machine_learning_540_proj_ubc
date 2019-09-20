
include("./random_smo.jl")
include("./smo_gsq_exact.jl")
include("./smo_gsq_approx_1.jl")
include("./smo_gsq_approx_2.jl")
include("./smo_gsq_approx_3.jl")
include("./smo_gsq_approx_4.jl")
include("./smo_gsq_approx_5.jl")

using Random
using LinearAlgebra
using Statistics
using Plots

function average_everything()
    # number of iterations
    max_iter = 5e2

    # set up averages for training loss
    trainErr_1_avg = zeros(Int(max_iter))
    trainErr_2_avg = zeros(Int(max_iter))
    trainErr_3_avg = zeros(Int(max_iter))
    trainErr_4_avg = zeros(Int(max_iter))
    trainErr_5_avg = zeros(Int(max_iter))
    trainErr_6_avg = zeros(Int(max_iter))
    trainErr_5a_avg = zeros(Int(max_iter))
    trainErr_5b_avg = zeros(Int(max_iter))
    trainErr_5c_avg = zeros(Int(max_iter))

    # set up averages for testing error
    testErr_1_avg = zeros(Int(max_iter))
    testErr_2_avg = zeros(Int(max_iter))
    testErr_3_avg = zeros(Int(max_iter))
    testErr_4_avg = zeros(Int(max_iter))
    testErr_5_avg = zeros(Int(max_iter))
    testErr_6_avg = zeros(Int(max_iter))
    testErr_5a_avg = zeros(Int(max_iter))
    testErr_5b_avg = zeros(Int(max_iter))
    testErr_5c_avg = zeros(Int(max_iter))

    # set averaging
    averaging = 100

    for i = 1:averaging

        # generate a new random set of data
        X_fake = rand(100,2)
        X_fake[1:50,:] = X_fake[1:50,:] - 1*rand(50,2)
        y_fake = ones(100)
        y_fake[1:50] = -1*ones(50)


        X_faket = rand(100,2)
        X_faket[1:50,:] = X_faket[1:50,:] - 1*rand(50,2)
        y_faket = ones(100)
        y_faket[1:50] = -1*ones(50)

        # hyper parameters
        C = 1.0
        epsilon = 0.001
        kernal_func = linear_kernal

        println("iteration: ", i)
        # # random
        trainErr_1, testErr_1, count_1, support_vectors_1 = fit_gsq_random(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false)
        trainErr_1_avg += trainErr_1
        testErr_1_avg += testErr_1
        println("random")
        println(count_1, ", ", support_vectors_1)
        # # exact
        trainErr_2, testErr_2, count_2, support_vectors_2 = fit_gsq_exact(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false)
        trainErr_2_avg += trainErr_2
        testErr_2_avg += testErr_2
        println("exact")
        println(count_2, ", ", support_vectors_2)
        # first approximation H = L * I
        trainErr_3, testErr_3, count_3, support_vectors_3 = fit_gsq_approx_1(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false)
        trainErr_3_avg += trainErr_3
        testErr_3_avg += testErr_3
        println("H = L*I")
        println(count_3, ", ", support_vectors_3)
        # # second
        trainErr_4, testErr_4, count_4, support_vectors_4 = fit_gsq_approx_2(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false)
        trainErr_4_avg += trainErr_4
        testErr_4_avg += testErr_4
        println("2 - step: choose max/min, then apply gs-q")
        println(count_4,", ", support_vectors_4)
        # third
        trainErr_5, testErr_5, count_5, support_vectors_5 = fit_gsq_approx_3(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false)
        trainErr_5_avg += trainErr_5
        testErr_5_avg += testErr_5
        println("2 - step: choose based on (f'_i - f'_j)^2 / (L[i] - L[j]) ")
        println(count_5,", ", support_vectors_5)
        # fourth
        trainErr_6, testErr_6, count_6, support_vectors_6 = fit_gsq_approx_4(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false)
        trainErr_6_avg += trainErr_6
        testErr_6_avg += testErr_6
        println("H = diag(H)")
        println(count_5, ", ", support_vectors_5)
        # sub-sampling GSf-q exact
        trainErr_5, testErr_5, count_5, support_vectors_5 = fit_gsq_approx_5(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false, 1000)
        trainErr_5a_avg += trainErr_5
        testErr_5a_avg += testErr_5
        println("random exact - 1000")
        println(count_5, support_vectors_5)
        trainErr_5, testErr_5, count_5, support_vectors_5 = fit_gsq_approx_5(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false, 500)
        trainErr_5b_avg += trainErr_5
        testErr_5b_avg += testErr_5
        println("random exact - 500")
        println(count_5, support_vectors_5)
        trainErr_5, testErr_5, count_5, support_vectors_5 = fit_gsq_approx_5(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false, 100)
        trainErr_5c_avg += trainErr_5
        testErr_5c_avg += testErr_5
        println("random exact - 100")
        println(count_5, support_vectors_5)
    end

    # now plot it all
    plot(1:max_iter, [testErr_1_avg, testErr_2_avg, testErr_3_avg, testErr_4_avg, testErr_5_avg, testErr_6_avg, testErr_5a_avg, testErr_5b_avg, testErr_5c_avg] ./ averaging,
                label=["GSr-q" "GSf-q" "GSl-q" "LibSVM" "graident diff" "GSd-q" "sample based GSf-q (nlogn)" "sample based GSf-q 1/2nlog(n)" "sample based GSf-q (n)"] )
    savefig("test_error.png")
    plot(1:max_iter, [trainErr_1_avg, trainErr_2_avg, trainErr_3_avg, trainErr_4_avg, trainErr_5_avg, trainErr_6_avg, trainErr_5a_avg, trainErr_5b_avg, trainErr_5c_avg] ./ averaging,
                label=["GSr-q" "GSf-q" "GSl-q" "LibSVM" "graident diff" "GSd-q" "sample based GSf-q (nlogn)" "sample based GSf-q 1/2nlog(n)" "sample based GSf-q (n)"])
    savefig("obj_value.png")

end
average_everything()
