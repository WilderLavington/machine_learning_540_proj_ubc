using LIBSVM
using RDatasets: dataset

function load_datasets(dataset_name)
    if dataset_name == "Participation"
        # predict if they are Foreign
        data = dataset("Ecdat", "Participation")
        # convert columns to -1, 1

        # convert dataframe to matrix
    elseif dataset_name == "schizophrenia2"
        # predict schizophrenia
        data = dataset("HSAUR", "schizophrenia2")
    elseif dataset_name == "womensrole"
        # predict sex
        data = dataset("HSAUR", "womensrole")
    elseif dataset_name == "kidrecurr"
        # predict indicator of infection (infect1)
        data = dataset("KMsurv", "kidrecurr")
    elseif dataset_name == "Vocab"
        # predict sex
        data = dataset("car", "Vocab")
    elseif dataset_name == "Hitters"
        # needs fiddling
        data = dataset("ISLR", "Hitters")
    elseif dataset_name == "Caravan"
        # needs fiddling
        data = dataset("ISLR", "Caravan")
    elseif dataset_name == "Credit"
        # needs fiddling
        data = dataset("ISLR", "Credit")
    elseif dataset_name == "VA"
        # needs fiddling
        data = dataset("MASS", "VA")
    elseif dataset_name == "SLID"
        # needs fiddling
        data = dataset("car", "SLID")
    elseif dataset_name == "mhtdata"
        # needs fiddling
        data = dataset("gap", "mhtdata")
    elseif dataset_name == "movies"
        # needs fiddling
        data = dataset("ggplot2", "movies")
    elseif dataset_name == "diamonds"
        # needs fiddling
        data = dataset("ggplot2", "diamonds")
    elseif dataset_name == "VerbAgg"
        # needs fiddling
        data = dataset("lme4", "VerbAgg")
    elseif dataset_name == "InstEval"
        # needs fiddling
        data = dataset("lme4", "InstEval")
    elseif dataset_name == "star"
        # needs fiddling
        data = dataset("mlmRev", "star")
    elseif dataset_name == "Chem97"
        # needs fiddling
        data = dataset("mlmRev", "Chem97")
    elseif dataset_name == "guImmun"
        # needs fiddling
        data = dataset("mlmRev", "guImmun")
    elseif dataset_name == "guPrenat"
        # needs fiddling
        data = dataset("mlmRev", "guPrenat")
    elseif dataset_name == "LaborSupply"
        # needs fiddling
        data = dataset("plm", "LaborSupply")
    elseif dataset_name == "baseball"
        # needs fiddling
        data = dataset("plyr", "baseball")
    elseif dataset_name == "presidentialElections"
        # needs fiddling
        data = dataset("pscl", "presidentialElections")
    elseif dataset_name == "admit"
        # needs fiddling
        data = dataset("pscl", "admit")
    elseif dataset_name == "uis"
        # needs fiddling
        data = dataset("quantreg", "uis")
    elseif dataset_name == "CrohnD"
        # needs fiddling
        data = dataset("robustbase", "CrohnD")
    elseif dataset_name == "NOxEmissions"
        # needs fiddling
        data = dataset("robustbase", "NOxEmissions")
    elseif dataset_name == "stagec"
        # needs fiddling
        data = dataset("rpart", "stagec")
    elseif dataset_name == "CNES"
        # needs fiddling
        data = dataset("sem", "CNES")
    elseif dataset_name == "colon"
        # needs fiddling
        data = dataset("survival", "colon")
    elseif dataset_name == "pbc"
        # needs fiddling
        data = dataset("survival", "pbc")
    elseif dataset_name == "nwtco"
        # needs fiddling
        data = dataset("survival", "nwtco")
    elseif dataset_name == "Bundesliga"
        # needs fiddling
        data = dataset("vcd", "Bundesliga")
    else
        println("please provide a valid dataset name")

    end
end


#Classification C-SVM
iris = dataset("datasets", "iris")
labels = convert(Vector, iris[:, :Species])
instances = convert(Array, iris[:, 1:4])
model = fit!(SVC(), instances[1:2:end, :], labels[1:2:end])
yp = predict(model, instances[2:2:end, :])
