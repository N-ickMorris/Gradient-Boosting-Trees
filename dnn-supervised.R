# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# set the path of where the input files are
mywd = "C:/Users/Nick Morris/Downloads/Data-Science/Model-Scripts/Supervised/Deep-Neural-Network"

# create a name for a .txt file to log progress information while parallel processing
myfile = "log.txt"

# -----------------------------------------------------------------------------------
# ---- Packages ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# data handling
require(data.table)
require(gtools)

# plotting
require(ggplot2)
require(scales)
require(GGally)

# modeling
require(forecast)
require(AlgDesign)
require(caret)
require(pROC)
require(h2o)
require(MLmetrics)

# parallel computing
require(foreach)
require(parallel)
require(doSNOW)

}

# -----------------------------------------------------------------------------------
# ---- Functions --------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# these are functions i like to use

# ---- prints the data types of each column in a data frame -------------------------

types = function(dat)
{
  # make dat into a data.frame
  dat = data.frame(dat)
  
  # get the column names
  column = sapply(1:ncol(dat), function(i) colnames(dat)[i])
  
  # get the class of the columns
  data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
  
  # compute the number of levels for each column
  levels = sapply(1:ncol(dat), function(i) ifelse(data.type[i] == "factor", length(levels(droplevels(dat[,i]))), 0))
  
  return(data.frame(column, data.type, levels))
}

# ---- builds a square confusion matrix ---------------------------------------------

confusion = function(ytrue, ypred)
{
  # load packages we need
  require(gtools)
  
  # make ypred and ytrue into factors, if they aren't already
  if(class(ytrue) != "factor") ytrue = factor(ytrue)
  if(class(ypred) != "factor") ypred = factor(ypred)
  
  # combine their levels into one unique set of levels
  common.levels = mixedsort(unique(c(levels(ytrue), levels(ypred))))
  
  # give each vector the same levels
  ytrue = factor(ytrue, levels = common.levels)
  ypred = factor(ypred, levels = common.levels)
  
  # return a square confusion matrix
  return(table("Actual" = ytrue, "Predicted" = ypred))
}

}

# -----------------------------------------------------------------------------------
# ---- Design Experiment ------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# set the work directory
setwd(mywd)

# create the file
file.create(myfile)

# read in the data
train = data.table(read.csv("train.csv"))
test = data.table(read.csv("test.csv"))
info = data.table(read.csv("input-supervised.csv", stringsAsFactors = FALSE))

# get the name of the response variable
Y.name = info[Metric == "response.variable.name", Value]

# get the data type of the response variable
Y.type = info[Metric == "response.variable.data.type", Value]

# get the portion of the data in train that will be used to build models when tuning the parameters
split = as.numeric(info[Metric == "partition", Value])

# get the number of times we will build each model
times = as.numeric(info[Metric == "times", Value])

# get the number for controling the stratified random sampling
seed = as.numeric(info[Metric == "seed", Value])

# get the proportion of cores that will be used for model building
cores = as.numeric(info[Metric == "cores", Value])

# remove any NA's in train and test
train = na.omit(train)
test = na.omit(test)

# update train to have just numeric predictor variables
train = cbind(unname(unlist(train[, Y.name, with = FALSE])), 
              data.table(model.matrix(~., data = train[, !Y.name, with = FALSE])[,-1]))

# update the column names of train
setnames(train, c(Y.name, names(train)[-1]))

# update test to have just numeric predictor variables
test = data.table(model.matrix(~., data = test)[,-1])

# get predictors (X) and response (Y)
X = as.matrix(train[,!Y.name, with = FALSE])
Y = as.character(unname(unlist(train[, Y.name, with = FALSE])))

# set up model IDs
mod.id = data.table(mod = 1)

# export mod.id
write.csv(mod.id, "dnn-model-ID-numbers.csv", row.names = FALSE)

# create a copy of Y for splitting the train into two groups: model training, model testing
if(Y.type != "Numeric")
{
  Y.split = as.factor(Y)
  
} else
{
  Y.split = as.numeric(Y)
}

# set the seed for reproducibility
set.seed(seed)

# create the partitions
data.split = createDataPartition(y = Y.split, 
                                 p = split,
                                 times = times)

# add a split column to doe
doe = data.table(split = 1:length(data.split))

# set up model parameters
if(Y.type == "Binary")
{
  # create a table to map the old levels of Y into the new levels
  Y.mapping = data.table(old.level = levels(Y.split),
                         new.level = 0:(length(levels(Y.split)) - 1))
  
  # reset the names of the levels of Y
  Y = as.numeric(Y.split) - 1
  
  # compute the number of instances across the 2 classes
  cases = table(Y)
  
  # determine the number of negative and positive cases
  cases = data.table(negative = cases[which(names(cases) == 0)],
                     positive = cases[which(names(cases) == 1)])
  
  # make Y into a factor
  Y = factor(Y, levels = 0:1)
  
  # the classes are imbalanced so set up the balance_classes and class_sampling_factors parameters
  balance_classes = TRUE
  class_sampling_factors = table(Y)
  class_sampling_factors = as.vector(max(class_sampling_factors) / class_sampling_factors)
  
} else if(Y.type == "Numeric")
{
  # update Y
  Y = Y.split
  
  # compute the number of classes
  num_class = 0
  
} else
{
  # create a table to map the old levels of Y into the new levels
  Y.mapping = data.table(old.level = levels(Y.split),
                         new.level = 0:(length(levels(Y.split)) - 1))
  
  # reset the names of the levels of Y
  Y = as.numeric(Y.split) - 1
  
  # compute the number of classes
  num_class = max(Y) + 1
  
  # make Y into a factor
  Y = factor(Y, levels = 0:(num_class - 1))
  
  # the classes are imbalanced so set up the balance_classes and class_sampling_factors parameters
  balance_classes = TRUE
  class_sampling_factors = table(Y)
  class_sampling_factors = as.vector(max(class_sampling_factors) / class_sampling_factors)
}

# set up epochs ~ a number indicating how many times the training data should be passed through the network to adjust path weights
# increasing this will increase computation time but this parameter is a big part of deep learning
# the default is 10, i read online that people will go up to 1000000
# by default the neural network uses early stopping, so if the model converges before reaching the value of epochs then the model stops
epochs = 10000

# choose the total number of hidden nodes per layer, the default is 200
# one rule of thumb for a single layer neural networks is 2/3 of the input layer size plus the output layer size
rule.of.thumb = round(((2/3) * ncol(X)) + max(1, length(levels(Y)) - 1), 0)

# here's the rule of thumb value
rule.of.thumb

# another technique i saw online for picking the number of nodes per layer is to try powers of 2 (ie. 2^n)
powers.of.two = 2^(1:10)
powers.of.two

# we are using epochs, adaptive learning rate, drop out ratios, and L1/L2 norm regularization
# so we can use more nodes than the rule of thumb because the above parameters are used to prevent overfitting
# this can be a vector, for example: c(100, 200) would mean that a 100 and 200 nodes per layer would be tested
# choose the total number of hidden nodes per layer
nodes = 2^10

# choose the number of hidden layers, the default is 2
# deep learning is expected to be done with multiple layers, and most applications use 2 or 3 layers
# this can be a vector, for example: 1:3 would mean that a single, double, and triple hidden layered networks would be built
# choose the number of hidden layers
layers = 2

# choose the number of folds
# nfolds + 1 models will be built:
# nfolds models with (1 / nfolds)-th of the data held out for predicting
# 1 model on the full training data using the combined predictions of the nfolds models
nfolds = 3

# since we have a classification problem with slightly imbalanced classes, we will use stratified random sampling
# otherwise you can set fold_assignment = "Random"
fold_assignment = "Stratified"

# remove objects we no longer need
rm(info, split, Y.split, times, seed, mywd)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Cross Validation -------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# build a function that will report prediction results of our model
dnn.pred = function(Xtrain, Ytrain, Xtest, Ytest, Y.type, cases, num_class, nfolds, fold_assignment, hidden, epochs, balance_classes, class_sampling_factors)
{
  # decide which type of model to run
  if(Y.type == "Binary")
  {
    # build the table for training the model
    dat = data.table(Xtrain)
    dat[, y := Ytrain]
    
    # make dat and Xtest into h2o objects
    dat.h2o = as.h2o(dat)
    Xtest.h2o = as.h2o(Xtest)
    
    # identify predictors (x) and response (y)
    y = "y"
    x = colnames(Xtrain)
    
    # build the training model
    # if the model runs into convergence issues then use the other activation parameter which is commented out
    mod = h2o.deeplearning(y = y,
                           x = x,
                           training_frame = dat.h2o,
                           nfolds = nfolds,
                           fold_assignment = fold_assignment,
                           hidden = hidden,
                           epochs = epochs,
                           balance_classes = balance_classes,
                           class_sampling_factors = class_sampling_factors,
                           l1 = 1e-5,
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = rep(0.5, length(hidden)),
                           activation = "RectifierWithDropout",
                           # activation = "TanhWithDropout",
                           variable_importances = FALSE,
                           seed = 42)
    
    # make predictions with the training model using the test set
    ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
    
    # free memory
    gc()
    
    # build the table for re-training the model
    dat2 = data.table(Xtest)
    dat2[, y := factor(as.numeric(ynew >= 0.5), levels = 0:1)]
    dat = rbind(dat, dat2)
    dat.h2o = as.h2o(dat)
    
    # build the training model
    # if the model runs into convergence issues then use the other activation parameter which is commented out
    mod = h2o.deeplearning(y = y,
                           x = x,
                           training_frame = dat.h2o,
                           nfolds = nfolds,
                           fold_assignment = fold_assignment,
                           hidden = hidden,
                           epochs = epochs,
                           balance_classes = balance_classes,
                           class_sampling_factors = class_sampling_factors,
                           l1 = 1e-5,
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = rep(0.5, length(hidden)),
                           activation = "RectifierWithDropout",
                           # activation = "TanhWithDropout",
                           variable_importances = FALSE,
                           seed = 42)
    
    # make predictions with the training model using the test set
    ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
    
    # free memory
    gc()
    
    # convert Ytest into a numeric variable
    Ytest = as.numeric(as.character(Ytest))
    
    # compute log loss
    Log.Loss = LogLoss(y_pred = ynew, 
                       y_true = Ytest)
    
    # build an ROC object for computing AUC and cutoff
    mod.roc = roc(Ytest ~ ynew)
    
    # compute the cut-off point that maximize accuracy
    cutoff = coords(mod.roc, x = "best", best.method = "closest.topleft", best.weights = c(1, cases$positive / sum(cases)))[1]
    
    # if the cuttoff is Inf or -Inf then set it to 0.5
    cutoff = ifelse(abs(cutoff) == Inf, 0.5, cutoff)
    
    # use the cutoff point to define predictions
    ynew = as.numeric(ynew >= cutoff)
    
    # compute a binary confusion matrix
    conf = confusion(ytrue = Ytest, ypred = ynew)
    
    # get the four cases from conf
    conf.cases = data.table(TP = conf[2,2], TN = conf[1,1], FP = conf[1,2], FN = conf[2,1])
    
    # build a table to summarize the performance of our training model
    output = data.table(Accuracy = (conf.cases$TP + conf.cases$TN) / (conf.cases$FP + conf.cases$TN + conf.cases$TP + conf.cases$FN),
                        Sensitivity = conf.cases$TP / (conf.cases$TP + conf.cases$FN),
                        Specificity = conf.cases$TN / (conf.cases$FP + conf.cases$TN),
                        AUC = as.numeric(auc(mod.roc)),
                        Odds.Ratio = (conf.cases$TP * conf.cases$TN) / (conf.cases$FN * conf.cases$FP),
                        Log.Loss = Log.Loss,
                        Cutoff = cutoff)
    
    # replace any NaN with NA
    output = as.matrix(output)
    output[is.nan(output)] = NA
    output = data.table(output) 
    
  } else if(Y.type == "Numeric")
  {
    # build the table for training the model
    dat = data.table(Xtrain)
    dat[, y := Ytrain]
    
    # make dat and Xtest into h2o objects
    dat.h2o = as.h2o(dat)
    Xtest.h2o = as.h2o(Xtest)
    
    # identify predictors (x) and response (y)
    y = "y"
    x = colnames(Xtrain)
    
    # build the training model
    # if the model runs into convergence issues then use the other activation parameter which is commented out
    mod = h2o.deeplearning(y = y,
                           x = x,
                           training_frame = dat.h2o,
                           nfolds = nfolds,
                           fold_assignment = fold_assignment,
                           hidden = hidden,
                           epochs = epochs,
                           l1 = 1e-5,
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = rep(0.5, length(hidden)),
                           activation = "RectifierWithDropout",
                           # activation = "TanhWithDropout",
                           variable_importances = FALSE,
                           seed = 42)
    
    # make predictions with the training model using the test set
    ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
    
    # free memory
    gc()
    
    # build the table for re-training the model
    dat2 = data.table(Xtest)
    dat2[, y := ynew]
    dat = rbind(dat, dat2)
    dat.h2o = as.h2o(dat)
    
    # build the training model
    # if the model runs into convergence issues then use the other activation parameter which is commented out
    mod = h2o.deeplearning(y = y,
                           x = x,
                           training_frame = dat.h2o,
                           nfolds = nfolds,
                           fold_assignment = fold_assignment,
                           hidden = hidden,
                           epochs = epochs,
                           l1 = 1e-5,
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = rep(0.5, length(hidden)),
                           activation = "RectifierWithDropout",
                           # activation = "TanhWithDropout",
                           variable_importances = FALSE,
                           seed = 42)
    
    # make predictions with the training model using the test set
    ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
    
    # free memory
    gc()
    
    # build a table to summarize the performance of our training model
    output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                              cor(ynew, Ytest)^2))
    
    setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
    
  } else
  {
    # build the table for training the model
    dat = data.table(Xtrain)
    dat[, y := Ytrain]
    
    # make dat and Xtest into h2o objects
    dat.h2o = as.h2o(dat)
    Xtest.h2o = as.h2o(Xtest)
    
    # identify predictors (x) and response (y)
    y = "y"
    x = colnames(Xtrain)
    
    # build the training model
    # if the model runs into convergence issues then use the other activation parameter which is commented out
    mod = h2o.deeplearning(y = y,
                           x = x,
                           training_frame = dat.h2o,
                           nfolds = nfolds,
                           fold_assignment = fold_assignment,
                           hidden = hidden,
                           epochs = epochs,
                           balance_classes = balance_classes,
                           class_sampling_factors = class_sampling_factors,
                           l1 = 1e-5,
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = rep(0.5, length(hidden)),
                           activation = "RectifierWithDropout",
                           # activation = "TanhWithDropout",
                           variable_importances = FALSE,
                           seed = 42)
    
    # make predictions with the training model
    ynew = as.data.frame(predict(mod, newdata = Xtest.h2o))
    
    # extract the probability prediction matrix
    ypred = as.matrix(ynew[,-1])
    
    # free memory
    gc()
    
    # build the table for re-training the model
    dat2 = data.table(Xtest)
    dat2[, y := factor(apply(ypred, 1, which.max) - 1, levels = 0:(num_class - 1))]
    dat = rbind(dat, dat2)
    dat.h2o = as.h2o(dat)
    
    # build the training model
    # if the model runs into convergence issues then use the other activation parameter which is commented out
    mod = h2o.deeplearning(y = y,
                           x = x,
                           training_frame = dat.h2o,
                           nfolds = nfolds,
                           fold_assignment = fold_assignment,
                           hidden = hidden,
                           epochs = epochs,
                           balance_classes = balance_classes,
                           class_sampling_factors = class_sampling_factors,
                           l1 = 1e-5,
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = rep(0.5, length(hidden)),
                           activation = "RectifierWithDropout",
                           # activation = "TanhWithDropout",
                           variable_importances = FALSE,
                           seed = 42)
    
    # make predictions with the training model
    ynew = as.data.frame(predict(mod, newdata = Xtest.h2o))
    
    # extract the probability prediction matrix
    ypred = as.matrix(ynew[,-1])
    
    # free memory
    gc()
    
    # ---- multi log loss metric ----
    
    # build a matrix indicating the true class values
    ytrue = model.matrix(~., data = data.frame(factor(Ytest, levels = 0:(num_class - 1))))[,-1]
    ytrue = cbind(1 - rowSums(ytrue), ytrue)
    
    # compute the multi-class logarithmic loss
    mll = MultiLogLoss(y_pred = ypred, y_true = ytrue)
    
    # ---- Kappa ----
    
    # get the predicted classes and actual classes
    ypred = apply(ypred, 1, which.max) - 1
    ytrue = factor(Ytest, levels = 0:(num_class - 1))
    
    # build a square confusion matrix
    conf = confusion(ytrue = ytrue, ypred = ypred)
    
    # get the total number of observations
    n = sum(conf)
    
    # get the vector of correct predictions
    dia = diag(conf)
    
    # get the vector of the number of observations per class
    rsum = rowSums(conf)
    
    # get the vector of the number of predictions per class
    csum = colSums(conf)
    
    # get the proportion of observations per class
    p = rsum / n
    
    # get the proportion of predcitions per class
    q = csum / n
    
    # compute accuracy
    acc = sum(dia) / n
    
    # compute expected accuracy
    exp.acc = sum(p * q)
    
    # compute kappa
    kap = (acc - exp.acc) / (1 - exp.acc)
    
    # ---- one-vs-all metrics ----
    
    # compute a binary confusion matrix for each class
    one.v.all = lapply(1:nrow(conf), function(i)
    {
      # get the four entries of a binary confusion matrix
      v = c(conf[i,i], 
            rsum[i] - conf[i,i], 
            csum[i] - conf[i,i], 
            n - rsum[i] - csum[i] + conf[i,i]);
      
      # build the confusion matrix
      return(matrix(v, nrow = 2, byrow = TRUE))
    })
    
    # sum up all of the matrices
    one.v.all = Reduce('+', one.v.all)
    
    # compute the micro average accuracy
    micro.acc = sum(diag(one.v.all)) / sum(one.v.all)
    
    # get the macro accuracy
    macro.acc = acc
    
    # combine all of our performance metrics
    output = data.table(Multi.Log.Loss = mll, Kappa = kap,
                        Macro.Accuracy = macro.acc, Micro.Accuracy = micro.acc)
  }
  
  return(output)
}

# choose the number of workers/threads and tasks for parallel processing (if you want to)
# specifying a value > 1 for workers means that multiple models in doe will be built in parallel
# specifying a value > 1 for nthread means that each model will internally be built in parallel
workers = 1
nthread = max(1, floor(cores * detectCores()))
tasks = nrow(doe)

# initialize the h2o instance
h2o.init(nthreads = nthread)

# remove any objects in the h2o instance
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# set up a cluster if workers > 1, otherwise don't set up a cluster
if(workers > 1)
{
  # setup parallel processing
  cl = makeCluster(workers, type = "SOCK", outfile = "")
  dnnisterDoSNOW(cl)
  
  # define %dopar%
  `%fun%` = `%dopar%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("deep neural network - cross validation\n")
  cat(paste(workers, "workers started at", Sys.time(), "\n"))
  sink()
  
} else
{
  # define %do%
  `%fun%` = `%do%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("deep neural network - cross validation\n")
  cat(paste("task 1 started at", Sys.time(), "\n"))
  sink()
}

# perform cross validation for each of the models in doe
dnn.cv = foreach(i = 1:tasks) %fun%
{
  # load packages we need for our tasks
  require(forecast)
  require(MLmetrics)
  require(data.table)
  require(h2o)
  require(pROC)

  # get the training and test sets
  rows = data.split[[doe$split[i]]]
  Xtrain = X[rows,]
  Ytrain = Y[rows]
  Xtest= X[-rows,]
  Ytest = Y[-rows]
  
  # build model and get prediction results
  output = dnn.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    hidden = rep(nodes, layers), epochs = epochs, nfolds = nfolds, 
                    fold_assignment = fold_assignment, balance_classes = balance_classes, 
                    class_sampling_factors = class_sampling_factors, num_class = num_class, 
                    Y.type = Y.type, cases = cases)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# end the cluster if it was set up
if(workers > 1)
{
  stopCluster(cl)
}

# combine the list of tables into one table
dnn.cv = rbindlist(dnn.cv)

}

# -----------------------------------------------------------------------------------
# ---- Export Results ---------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# export dnn.cv
write.csv(dnn.cv, "dnn-cross-validation-performance.csv", row.names = FALSE)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

# get the key columns from mod.id
keys = names(mod.id[,!"mod"])

# summarize each of the performance metrics
# decide which type of model to run
if(Y.type == "Binary")
{
  # compute diagnostic errors
  dnn.diag = dnn.cv[,.(stat = factor(stat, levels = stat),
                       Accuracy = as.vector(summary(na.omit(Accuracy))), 
                       Sensitivity = as.vector(summary(na.omit(Sensitivity))),
                       Specificity = as.vector(summary(na.omit(Specificity))),
                       AUC = as.vector(summary(na.omit(AUC))),
                       Odds.Ratio = as.vector(summary(na.omit(Odds.Ratio))),
                       Log.Loss = as.vector(summary(na.omit(Log.Loss))),
                       Cutoff = as.vector(summary(na.omit(Cutoff)))),
                    by = keys]
  
} else if(Y.type == "Numeric")
{
  # compute diagnostic errors
  dnn.diag = dnn.cv[,.(stat = factor(stat, levels = stat),
                       ME = as.vector(summary(na.omit(ME))), 
                       RMSE = as.vector(summary(na.omit(RMSE))),
                       MAPE = as.vector(summary(na.omit(MAPE))),
                       Rsquared = as.vector(summary(na.omit(Rsquared)))),
                    by = keys]
  
  
} else
{
  # compute diagnostic errors
  dnn.diag = dnn.cv[,.(stat = factor(stat, levels = stat),
                       Multi.Log.Loss = as.vector(summary(na.omit(Multi.Log.Loss))),
                       Kappa = as.vector(summary(na.omit(Kappa))),
                       Macro.Accuracy = as.vector(summary(na.omit(Macro.Accuracy))),
                       Micro.Accuracy = as.vector(summary(na.omit(Micro.Accuracy)))),
                    by = keys]
}

# add a mod column to dnn.diag
if(length(keys) > 0)
{
  # set the keys in mod.id and dnn.diag
  setkeyv(mod.id, keys)
  setkeyv(dnn.diag, keys)
  
  # join mod.id onto dnn.diag
  dnn.diag = mod.id[dnn.diag]
  
} else
{
  dnn.diag[, mod := 1]
}

# lets rank each model
# decide which type of model to run
if(Y.type == "Binary")
{
  # extract the median AUC & Log Loss of each model in dnn.diag
  dnn.rank = data.table(dnn.diag[stat == "Median", .(mod, AUC, Log.Loss)])
  
  # rescale AUC and Log Loss to a [0, 1] range
  dnn.rank[, AUC := rescale(AUC, to = c(0, 1))]
  dnn.rank[, Log.Loss := 1 - rescale(Log.Loss, to = c(0, 1))]
  
  # compute the model score
  dnn.rank[, score := AUC + Log.Loss]
  
  # sort dnn.rank by score
  dnn.rank = dnn.rank[order(-score)]
  
  # define the model rank
  dnn.rank[, rank := 1:nrow(dnn.rank)]
  
  # just keep mod and rank in dnn.rank
  dnn.rank = dnn.rank[,.(mod, rank)]
  
} else if(Y.type == "Numeric")
{
  # extract the median Rsquared & RMSE of each model in dnn.diag
  dnn.rank = data.table(dnn.diag[stat == "Median", .(mod, Rsquared, RMSE)])
  
  # rescale Rsquared and RMSE to a [0, 1] range
  dnn.rank[, Rsquared := rescale(Rsquared, to = c(0, 1))]
  dnn.rank[, RMSE := 1 - rescale(RMSE, to = c(0, 1))]
  
  # compute the model score
  dnn.rank[, score := Rsquared + RMSE]
  
  # sort dnn.rank by score
  dnn.rank = dnn.rank[order(-score)]
  
  # define the model rank
  dnn.rank[, rank := 1:nrow(dnn.rank)]
  
  # just keep mod and rank in dnn.rank
  dnn.rank = dnn.rank[,.(mod, rank)]
  
} else
{
  # extract the median Micro.Accuracy & Multi.Log.Loss of each model in dnn.diag
  dnn.rank = data.table(dnn.diag[stat == "Median", .(mod, Micro.Accuracy, Multi.Log.Loss)])
  
  # rescale Micro.Accuracy and Multi.Log.Loss to a [0, 1] range
  dnn.rank[, Micro.Accuracy := rescale(Micro.Accuracy, to = c(0, 1))]
  dnn.rank[, Multi.Log.Loss := 1 - rescale(Multi.Log.Loss, to = c(0, 1))]
  
  # compute the model score
  dnn.rank[, score := Micro.Accuracy + Multi.Log.Loss]
  
  # sort dnn.rank by score
  dnn.rank = dnn.rank[order(-score)]
  
  # define the model rank
  dnn.rank[, rank := 1:nrow(dnn.rank)]
  
  # just keep mod and rank in dnn.rank
  dnn.rank = dnn.rank[,.(mod, rank)]
}

# make mod the key column in dnn.rank and dnn.diag
setkey(dnn.diag, mod)
setkey(dnn.rank, mod)

# join dnn.rank onto dnn.diag
dnn.diag = dnn.rank[dnn.diag]

# order dnn.diag by rank
dnn.diag = dnn.diag[order(rank)]

# export dnn.diag
write.csv(dnn.diag, "dnn-summary.csv", row.names = FALSE)

# add rank to dnn.cv
dnn.cv[, rank := dnn.rank$rank]

# add a Model identity column
dnn.cv[, Model := factor(ifelse(rank == 1, "Yes", "No"), levels = c("Yes", "No"))]

# plot our models performance metrics
# decide which type of model to run
if(Y.type == "Binary")
{
  dnn.plot = ggpairs(dnn.cv[,.(Accuracy, AUC, Log.Loss, Model)],
                     mapping = aes(color = Model, fill = Model),
                     upper = list(continuous = wrap("density", alpha = 1/4), combo = "box"),
                     lower = list(continuous = wrap("points"), combo = wrap("dot")),
                     diag = list(continuous = wrap("densityDiag", alpha = 1/3))) + 
    theme_bw(base_size = 20)
  
} else if(Y.type == "Numeric")
{
  dnn.plot = ggpairs(dnn.cv[,.(Rsquared, RMSE, MAPE, Model)],
                     mapping = aes(color = Model, fill = Model),
                     upper = list(continuous = wrap("density", alpha = 1/4), combo = "box"),
                     lower = list(continuous = wrap("points"), combo = wrap("dot")),
                     diag = list(continuous = wrap("densityDiag", alpha = 1/3))) + 
    theme_bw(base_size = 20)
  
} else
{
  dnn.plot = ggpairs(dnn.cv[,.(Macro.Accuracy, Micro.Accuracy, Multi.Log.Loss, Model)],
                     mapping = aes(color = Model, fill = Model),
                     upper = list(continuous = wrap("density", alpha = 1/4), combo = "box"),
                     lower = list(continuous = wrap("points"), combo = wrap("dot")),
                     diag = list(continuous = wrap("densityDiag", alpha = 1/3))) + 
    theme_bw(base_size = 20)
}

# set up a graphics window
# windows()

# set up a pdf file to capture the next graphic
pdf("dnn-plot.pdf", width = 16, height = 10, paper = "special") 

# call the plot
print(dnn.plot)

# close off the connection
dev.off()

# extract the model parameters of the best model
dnn.best = data.table(mod.id[mod == dnn.rank[rank == 1, mod]])

# compute the hidden layers for the model
hidden = rep(nodes, layers)

# initialize the h2o instance
h2o.init(nthreads = nthread)

# remove the progress bar when model building
h2o.no_progress()

# predict the test set with the best model
# decide which type of model to run
if(Y.type == "Binary")
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # make dat and Xtest into h2o objects
  dat.h2o = as.h2o(dat)
  Xtest.h2o = as.h2o(test)
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  # if the model runs into convergence issues then use the other activation parameter which is commented out
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         nfolds = nfolds,
                         fold_assignment = fold_assignment,
                         hidden = hidden,
                         epochs = epochs,
                         balance_classes = balance_classes,
                         class_sampling_factors = class_sampling_factors,
                         l1 = 1e-5,
                         input_dropout_ratio = 0.2,
                         hidden_dropout_ratios = rep(0.5, length(hidden)),
                         activation = "RectifierWithDropout",
                         # activation = "TanhWithDropout",
                         seed = 42)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
  
  # make predictions with the model on the test set
  ynew = as.numeric(ynew >= dnn.diag[mod == dnn.best$mod & stat == "Median", Cutoff])
  
  # free memory
  gc()
  
  # build the table for re-training the model
  dat2 = data.table(test)
  dat2[, y := factor(ynew, levels = 0:1)]
  dat = rbind(dat, dat2)
  dat.h2o = as.h2o(dat)
  
  # build the training model
  # if the model runs into convergence issues then use the other activation parameter which is commented out
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         nfolds = nfolds,
                         fold_assignment = fold_assignment,
                         hidden = hidden,
                         epochs = epochs,
                         balance_classes = balance_classes,
                         class_sampling_factors = class_sampling_factors,
                         l1 = 1e-5,
                         input_dropout_ratio = 0.2,
                         hidden_dropout_ratios = rep(0.5, length(hidden)),
                         activation = "RectifierWithDropout",
                         # activation = "TanhWithDropout",
                         variable_importances = TRUE,
                         seed = 42)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
  
  # make predictions with the model on the test set
  ynew = as.numeric(ynew >= dnn.diag[mod == dnn.best$mod & stat == "Median", Cutoff])
  
  # free memory
  gc()
  
  # give ynew an ID column to preserve its order
  ynew = data.table(ID = 1:length(ynew),
                    value = ynew)
  
  # set value as key column in ynew, and new.level as the key column in Y.mapping
  setkey(ynew, value)
  setkey(Y.mapping, new.level)
  
  # join Y.mapping onto ynew
  ynew = Y.mapping[ynew]
  
  # order ynew by ID
  ynew = ynew[order(ID)]
  
  # add the predictions in ynew to test
  test[, Predictions := ynew$old.level]
  
} else if(Y.type == "Numeric")
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # make dat and Xtest into h2o objects
  dat.h2o = as.h2o(dat)
  Xtest.h2o = as.h2o(test)
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  # if the model runs into convergence issues then use the other activation parameter which is commented out
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         nfolds = nfolds,
                         fold_assignment = fold_assignment,
                         hidden = hidden,
                         epochs = epochs,
                         l1 = 1e-5,
                         input_dropout_ratio = 0.2,
                         hidden_dropout_ratios = rep(0.5, length(hidden)),
                         activation = "RectifierWithDropout",
                         # activation = "TanhWithDropout",
                         seed = 42)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
  
  # free memory
  gc()
  
  # build the table for re-training the model
  dat2 = data.table(test)
  dat2[, y := ynew]
  dat = rbind(dat, dat2)
  dat.h2o = as.h2o(dat)
  
  # build the training model
  # if the model runs into convergence issues then use the other activation parameter which is commented out
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         nfolds = nfolds,
                         fold_assignment = fold_assignment,
                         hidden = hidden,
                         epochs = epochs,
                         l1 = 1e-5,
                         input_dropout_ratio = 0.2,
                         hidden_dropout_ratios = rep(0.5, length(hidden)),
                         activation = "RectifierWithDropout",
                         # activation = "TanhWithDropout",
                         variable_importances = TRUE,
                         seed = 42)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(as.character(as.data.frame(predict(mod, newdata = Xtest.h2o))$predict))
  
  # free memory
  gc()
  
  # add the predictions in ynew to test
  test[, Predictions := ynew]
  
} else
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # make dat and Xtest into h2o objects
  dat.h2o = as.h2o(dat)
  Xtest.h2o = as.h2o(test)
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  # if the model runs into convergence issues then use the other activation parameter which is commented out
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         nfolds = nfolds,
                         fold_assignment = fold_assignment,
                         hidden = hidden,
                         epochs = epochs,
                         balance_classes = balance_classes,
                         class_sampling_factors = class_sampling_factors,
                         l1 = 1e-5,
                         input_dropout_ratio = 0.2,
                         hidden_dropout_ratios = rep(0.5, length(hidden)),
                         activation = "RectifierWithDropout",
                         # activation = "TanhWithDropout",
                         seed = 42)
  
  # make predictions with the training model
  ynew = as.data.frame(predict(mod, newdata = Xtest.h2o))
  
  # extract the probability prediction matrix
  ynew = as.matrix(ynew[,-1])
  
  # extract the class with the highest probability as the prediction
  ynew = apply(X = ynew, MARGIN = 1, FUN = which.max) - 1
  
  # free memory
  gc()
  
  # build the table for re-training the model
  dat2 = data.table(test)
  dat2[, y := factor(ynew, levels = 0:(num_class - 1))]
  dat = rbind(dat, dat2)
  dat.h2o = as.h2o(dat)
  
  # build the training model
  # if the model runs into convergence issues then use the other activation parameter which is commented out
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         nfolds = nfolds,
                         fold_assignment = fold_assignment,
                         hidden = hidden,
                         epochs = epochs,
                         balance_classes = balance_classes,
                         class_sampling_factors = class_sampling_factors,
                         l1 = 1e-5,
                         input_dropout_ratio = 0.2,
                         hidden_dropout_ratios = rep(0.5, length(hidden)),
                         activation = "RectifierWithDropout",
                         # activation = "TanhWithDropout",
                         variable_importances = TRUE,
                         seed = 42)
  
  # make predictions with the training model
  ynew = as.data.frame(predict(mod, newdata = Xtest.h2o))
  
  # extract the probability prediction matrix
  ynew = as.matrix(ynew[,-1])
  
  # extract the class with the highest probability as the prediction
  ynew = apply(X = ynew, MARGIN = 1, FUN = which.max) - 1
  
  # free memory
  gc()
  
  # give ynew an ID column to preserve its order
  ynew = data.table(ID = 1:length(ynew),
                    value = ynew)
  
  # set value as key column in ynew, and new.level as the key column in Y.mapping
  setkey(ynew, value)
  setkey(Y.mapping, new.level)
  
  # join Y.mapping onto ynew
  ynew = Y.mapping[ynew]
  
  # order ynew by ID
  ynew = ynew[order(ID)]
  
  # add the predictions in ynew to test
  test[, Predictions := ynew$old.level]
}

# extract variable importance
var.data = data.table(h2o.varimp(mod))

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# export test
write.csv(test, "dnn-test-predictions.csv", row.names = FALSE)

# export var.data
write.csv(var.data, "dnn-variable-data.csv", row.names = FALSE)

# free memory
gc()

}























