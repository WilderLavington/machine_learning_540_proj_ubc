
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

function data_set_1()

    # generate a new random set of data
    X_fake = rand(100,2)
    X_fake[1:50,:] = X_fake[1:50,:] - 1*rand(50,2)
    y_fake = ones(100)
    y_fake[1:50] = -1*ones(50)

    # use
    X_faket = rand(100,2)
    X_faket[1:50,:] = X_faket[1:50,:] - 1*rand(50,2)
    y_faket = ones(100)
    y_faket[1:50] = -1*ones(50)

    # return it all
    return X_fake, y_fake, X_faket, y_faket
end

function data_set_2()
    # generate a new random set of data
    X_fake = rand(500,2)
    X_fake[1:250,:] = X_fake[1:250,:] - 1*rand(250,2)
    y_fake = ones(500)
    y_fake[1:250] = -1*ones(250)

    # use
    X_faket = rand(500,2)
    X_faket[1:250,:] = X_faket[1:250,:] - 1*rand(250,2)
    y_faket = ones(500)
    y_faket[1:250] = -1*ones(250)

    # return it all
    return X_fake, y_fake, X_faket, y_faket
end

function data_set_3()
    # generate a new random set of data
    X_fake = rand(1000,2)
    X_fake[1:500,:] = X_fake[1:500,:] - 1*rand(500,2)
    y_fake = ones(1000)
    y_fake[1:500] = -1*ones(500)

    # use
    X_faket = rand(1000,2)
    X_faket[1:500,:] = X_faket[1:500,:] - 1*rand(500,2)
    y_faket = ones(1000)
    y_faket[1:500] = -1*ones(500)

    # return it all
    return X_fake, y_fake, X_faket, y_faket
end

function test(X_train, y_train, X_test, y_test, max_iter, averaging, obj_file_name, test_file_name)

    # hyper parameters
    C = 1.0
    epsilon = 0.001
    kernal_func = linear_kernal

    # set up averages for training loss
    Objective_evaluations = zeros(9, Int(max_iter))
    Test_errors = zeros(9, Int(max_iter))
    KKT_conditions_iterations = zeros(9)
    sufficient_convergence_iterations = zeros(9)

    # average over the number of iterations to convergence
    for i = 1:averaging
        println("iteration: ", i)

        # random
        print("random selection: ")
        trainErr_1, testErr_1, count_1, support_vectors_1 = fit_gsq_random(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false)
        Objective_evaluations[1,:] += trainErr_1
        Test_errors[1,:] += testErr_1
        KKT_conditions_iterations[1] += count_1

        # GSF-q
        print("GSF-q: ")
        trainErr_2, testErr_2, count_2, support_vectors_2 = fit_gsq_exact(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false)
        Objective_evaluations[2,:] += trainErr_2
        Test_errors[2,:] += testErr_2
        KKT_conditions_iterations[2] += count_2

        # H = L * I
        print("GSL-q: ")
        trainErr_3, testErr_3, count_3, support_vectors_3 = fit_gsq_approx_1(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false)
        Objective_evaluations[3,:] += trainErr_3
        Test_errors[3,:] += testErr_3
        KKT_conditions_iterations[3] += count_3

        # maximum gradient difference between
        print("LibSVM: ")
        trainErr_4, testErr_4, count_4, support_vectors_4 = fit_gsq_approx_2(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false)
        Objective_evaluations[4,:] += trainErr_4
        Test_errors[4,:] += testErr_4
        KKT_conditions_iterations[4] += count_4

        # heuristic
        print("Gradient Difference Heuristic: ")
        trainErr_5, testErr_5, count_5, support_vectors_5 = fit_gsq_approx_3(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false)
        Objective_evaluations[5,:] += trainErr_5
        Test_errors[5,:] += testErr_5
        KKT_conditions_iterations[5] += count_5

        # GSd-q
        print("GSD-q: ")
        trainErr_6, testErr_6, count_6, support_vectors_6 = fit_gsq_approx_4(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false)
        Objective_evaluations[6,:] += trainErr_6
        Test_errors[6,:] += testErr_6
        KKT_conditions_iterations[6] += count_6

        # sample based gsf n log n
        print("nlogn sample based GSF-q: ")
        trainErr_7, testErr_7, count_7, support_vectors_7 = fit_gsq_approx_5(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false, 1000)
        Objective_evaluations[7,:] += trainErr_7
        Test_errors[7,:] += testErr_7
        KKT_conditions_iterations[7] += count_7

        # sample based gsf 1/2 n log n
        print("1/2*nlogn sample based GSF-q: ")
        trainErr_8, testErr_8, count_8, support_vectors_8 = fit_gsq_approx_5(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false, 500)
        Objective_evaluations[8,:] += trainErr_8
        Test_errors[8,:] += testErr_8
        KKT_conditions_iterations[8] += count_8

        # sample based gsf n
        print("n sample based GSF-q: ")
        trainErr_9, testErr_9, count_9, support_vectors_9 = fit_gsq_approx_5(X_train, y_train, X_test, y_test, kernal_func, C, epsilon, max_iter, false, 100)
        Objective_evaluations[9,:] += trainErr_9
        Test_errors[9,:] += testErr_9
        KKT_conditions_iterations[9] += count_9

        # sample based gsf n
        # trainErr_10, testErr_10, count_10, support_vectors_10 = fit_gsq_approx_6(X_fake, y_fake, X_faket, y_faket, kernal_func, C, epsilon, max_iter, false, 100)
        # Objective_evaluations[10,:] += trainErr_10
        # Test_errors[10,:] += testErr_10
        # KKT_conditions_iterations[10,:] += count_10

    end

    # average everything
    Objective_evaluations = Objective_evaluations./ averaging
    Test_errors = Test_errors./ averaging
    KKT_conditions_iterations = KKT_conditions_iterations./ averaging

    # set of methods to be tested
    methods = ["GSr-q" "GSf-q" "GSl-q" "LibSVM" "graident diff" "GSd-q" "sample based GSf-q (nlogn)" "sample based GSf-q 1/2nlog(n)" "sample based GSf-q (n)"]

    # now plot it all then save it
    plot(1:max_iter,  Objective_evaluations', label=methods)
    savefig(obj_file_name)
    plot(1:max_iter, Test_errors', label=methods)
    savefig(test_file_name)

    # print the iterations
    for i=1:9
        println(methods[i], ",  ", KKT_conditions_iterations[i])
    end

    # nothing to return
    return 0
end


averaging = 10
max_iter = 1e3

# test 1:
X_train, y_train, X_test, y_test = data_set_1()
obj_file_name = "obj_value_1.png"
test_file_name = "test_error_1.png"
test(X_train, y_train, X_test, y_test, max_iter, averaging, obj_file_name, test_file_name)

# test 2:
X_train, y_train, X_test, y_test = data_set_2()
obj_file_name = "obj_value_2.png"
test_file_name = "test_error_2.png"
test(X_train, y_train, X_test, y_test, max_iter, averaging, obj_file_name, test_file_name)

# test 3
X_train, y_train, X_test, y_test = data_set_3()
obj_file_name = "obj_value_3.png"
test_file_name = "test_error_3.png"
test(X_train, y_train, X_test, y_test, max_iter, averaging, obj_file_name, test_file_name)
