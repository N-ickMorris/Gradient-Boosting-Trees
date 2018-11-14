# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# set the path of where the input files are
mywd = "C:/Users/Nick Morris/Downloads/Data-Science/Model-Scripts/Supervised/Gradient-Boosting-Trees"

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
require(pROC)
require(caret)
require(xgboost)
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

# we have 7 parameters of interest:
  # nrounds ~ the max number of boosting iterations
  # eta ~ the learning rate
  # max_depth ~ maximum depth of a tree
  # min_child_weight ~ minimum sum of instance weight needed in a child
  # subsample ~ the proportion of data (rows) to randomly sample each round
  # colsample_bytree ~ the proportion of variables (columns) to randomly sample each round
  # gamma ~ minimum loss reduction required to make a further partition on a leaf node of the tree

# create parameter combinations to test
doe = data.table(expand.grid(nrounds = c(50, 250),
                             eta = c(0.015, 0.025, 0.05, 0.1),
                             max_depth = c(5, 10, 20), 
                             min_child_weight = c(1, 5, 11),
                             subsample = c(0.7, 1),
                             colsample_bytree = c(0.7, 1),
                             gamma = c(0, 0.01, 0.5)))

# lets get a smaller optimal design from doe
# compute the number of levels for each variable in our design
levels.design2 = sapply(1:ncol(doe), function(j) length(table(doe[, j, with = FALSE])))

# build the general factorial design
doe.gen = gen.factorial(levels.design)

# compute a smaller optimal design
set.seed(42)
doe.opt = optFederov(data = doe.gen, 
                     nTrials = 2^8)

# update which rows to keep in doe according to doe.opt
doe = doe[doe.opt$rows]

# set up model IDs
# make a copy of doe
mod.id = data.table(doe)

# give mod.id an id column
mod.id[, mod := 1:nrow(mod.id)]

# export mod.id
write.csv(mod.id, "gbm-model-ID-numbers.csv", row.names = FALSE)

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
doe = rbindlist(lapply(1:length(data.split), function(i) 
  cbind(split = rep(i, nrow(doe)), 
        doe)))

# set up model parameters
if(Y.type == "Binary")
{
  # set the objective funtion
  objective = "binary:logistic"
  
  # set the performance metric
  eval_metric = "error"
  
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
  
  # compute a ratio to indicate the imbalance between classes
  scale.pos.weight = cases$negative / cases$positive
  
} else if(Y.type == "Numeric")
{
  # set the objective funtion
  objective = "reg:linear"
  
  # set the performance metric
  eval_metric = "rmse"
  
  # update Y
  Y = Y.split
  
  # compute the number of classes
  num_class = 0
  
} else
{
  # set the objective funtion
  objective = "multi:softprob"
  
  # set the performance metric
  eval_metric = "merror"
  
  # create a table to map the old levels of Y into the new levels
  Y.mapping = data.table(old.level = levels(Y.split),
                         new.level = 0:(length(levels(Y.split)) - 1))
  
  # reset the names of the levels of Y
  Y = as.numeric(Y.split) - 1
  
  # compute the number of classes
  num_class = max(Y) + 1
  
  # the classes are imbalanced so lets define the weight parameter where a class with more observations is weighted less
  weight = table(Y)
  weight = max(weight) / weight
  
  # set up weight into a table for joining
  weight = data.table(level = as.numeric(names(weight)), 
                      value = as.numeric(weight))
  
  # set up Y into a table for joining
  Y.table = data.table(ID = 1:length(Y),
                       level = Y)
  
  # set level as the key column in weight and Y.table
  setkey(Y.table, level)
  setkey(weight, level)
  
  # join weight onto Y.table
  weight = data.table(weight[Y.table])
  
  # order weight by ID and get the value column of weight
  weight = weight[order(ID)]
  weight = weight$value
}

# remove objects we no longer need
rm(doe.gen, doe.opt, info, levels.design, split, Y.split, times, seed, mywd)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Cross Validation -------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# build a function that will report prediction results of our model
gbm.pred = function(Xtrain, Ytrain, Xtest, Ytest, Y.type, cases, scale.pos.weight = NULL, weight = NULL, objective, eval_metric, eta, max_depth, nthread, nrounds, min_child_weight, gamma, verbose, subsample, colsample_bytree, num_class)
{
  # decide which type of model to run
  if(Y.type == "Binary")
  {
    # build the training model
    set.seed(42)
    mod = xgboost(label = Ytrain, data = Xtrain,
                  objective = objective, eval_metric = eval_metric,
                  eta = eta, max_depth = max_depth, nthread = nthread,
                  nrounds = nrounds, min_child_weight = min_child_weight,
                  gamma = gamma, verbose = verbose, scale.pos.weight = scale.pos.weight,
                  subsample = subsample, colsample_bytree = colsample_bytree)
  
    # make predictions with the training model using the test set
    ynew = as.numeric(predict(mod, newdata = Xtest))
    
    # free memory
    gc()
    
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
    # build the training model
    set.seed(42)
    mod = xgboost(label = Ytrain, data = Xtrain,
                  objective = objective, eval_metric = eval_metric,
                  eta = eta, max_depth = max_depth, nthread = nthread,
                  nrounds = nrounds, min_child_weight = min_child_weight,
                  gamma = gamma, verbose = verbose, subsample = subsample, 
                  colsample_bytree = colsample_bytree)
    
    # make predictions with the training model using the test set
    ynew = as.numeric(predict(mod, newdata = Xtest))
    
    # free memory
    gc()
    
    # build a table to summarize the performance of our training model
    output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                              cor(ynew, Ytest)^2))
    
    setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
    
  } else
  {
    # build the training model
    set.seed(42)
    mod = xgboost(label = Ytrain, data = Xtrain, num_class = num_class,
                  objective = objective, eval_metric = eval_metric,
                  eta = eta, max_depth = max_depth, nthread = nthread,
                  nrounds = nrounds, min_child_weight = min_child_weight,
                  gamma = gamma, verbose = verbose, weight = weight,
                  subsample = subsample, colsample_bytree = colsample_bytree)
    
    # get the probability prediction matrix
    ypred = predict(mod, newdata = Xtest, reshape = TRUE)
    
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
workers = max(1, floor(cores * detectCores()))
nthread = 1
tasks = nrow(doe)

# set up a cluster if workers > 1, otherwise don't set up a cluster
if(workers > 1)
{
  # setup parallel processing
  cl = makeCluster(workers, type = "SOCK", outfile = "")
  registerDoSNOW(cl)
  
  # define %dopar%
  `%fun%` = `%dopar%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("gradient boosting - cross validation\n")
  cat(paste(workers, "workers started at", Sys.time(), "\n"))
  sink()
  
} else
{
  # define %do%
  `%fun%` = `%do%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("gradient boosting - cross validation\n")
  cat(paste("task 1 started at", Sys.time(), "\n"))
  sink()
}

# perform cross validation for each of the models in doe
gbm.cv = foreach(i = 1:tasks) %fun%
{
  # load packages we need for our tasks
  require(forecast)
  require(MLmetrics)
  require(data.table)
  require(xgboost)
  require(pROC)

  # get the training and test sets
  rows = data.split[[doe$split[i]]]
  Xtrain = X[rows,]
  Ytrain = Y[rows]
  Xtest= X[-rows,]
  Ytest = Y[-rows]
  
  # decide which type of model to run
  if(Y.type == "Binary")
  {
    # build model and get prediction results
    output = gbm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, num_class = num_class,
                      objective = objective, eval_metric = eval_metric, nthread = nthread,
                      eta = doe$eta[i], max_depth = doe$max_depth[i], nrounds = doe$nrounds[i], 
                      min_child_weight = doe$min_child_weight[i], gamma = doe$gamma[i], verbose = 0,
                      subsample = doe$subsample[i], colsample_bytree = doe$colsample_bytree[i],
                      scale.pos.weight = scale.pos.weight, Y.type = Y.type, cases = cases)
    
  } else if(Y.type == "Numeric")
  {
    # build model and get prediction results
    output = gbm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, num_class = num_class,
                      objective = objective, eval_metric = eval_metric, nthread = nthread,
                      eta = doe$eta[i], max_depth = doe$max_depth[i], nrounds = doe$nrounds[i], 
                      min_child_weight = doe$min_child_weight[i], gamma = doe$gamma[i], verbose = 0,
                      subsample = doe$subsample[i], colsample_bytree = doe$colsample_bytree[i], Y.type = Y.type)
    
  } else
  {
    # get the weights of the training set
    weight.train = weight[rows]
    
    # build model and get prediction results
    output = gbm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, num_class = num_class,
                      objective = objective, eval_metric = eval_metric, nthread = nthread,
                      eta = doe$eta[i], max_depth = doe$max_depth[i], nrounds = doe$nrounds[i], 
                      min_child_weight = doe$min_child_weight[i], gamma = doe$gamma[i], verbose = 0,
                      subsample = doe$subsample[i], colsample_bytree = doe$colsample_bytree[i],
                      weight = weight.train, Y.type = Y.type)
  }
  
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

# end the cluster if it was set up
if(workers > 1)
{
  stopCluster(cl)
}

# combine the list of tables into one table
gbm.cv = rbindlist(gbm.cv)

}

# -----------------------------------------------------------------------------------
# ---- Export Results ---------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# export gbm.cv
write.csv(gbm.cv, "gbm-cross-validation-performance.csv", row.names = FALSE)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

# get the key columns from mod.id
keys = names(mod.id[,!"mod"])

# summarize each of the performance metrics
# decide which type of model to run
if(Y.type == "Binary")
{
  # compute diagnostic errors
  gbm.diag = gbm.cv[,.(stat = factor(stat, levels = stat),
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
  gbm.diag = gbm.cv[,.(stat = factor(stat, levels = stat),
                       ME = as.vector(summary(na.omit(ME))), 
                       RMSE = as.vector(summary(na.omit(RMSE))),
                       MAPE = as.vector(summary(na.omit(MAPE))),
                       Rsquared = as.vector(summary(na.omit(Rsquared)))),
                    by = keys]
  
  
} else
{
  # compute diagnostic errors
  gbm.diag = gbm.cv[,.(stat = factor(stat, levels = stat),
                       Multi.Log.Loss = as.vector(summary(na.omit(Multi.Log.Loss))),
                       Kappa = as.vector(summary(na.omit(Kappa))),
                       Macro.Accuracy = as.vector(summary(na.omit(Macro.Accuracy))),
                       Micro.Accuracy = as.vector(summary(na.omit(Micro.Accuracy)))),
                    by = keys]
}

# set the keys in mod.id and gbm.diag
setkeyv(mod.id, keys)
setkeyv(gbm.diag, keys)

# join mod.id onto gbm.diag
gbm.diag = mod.id[gbm.diag]

# lets rank each model
# decide which type of model to run
if(Y.type == "Binary")
{
  # extract the median AUC & Log Loss of each model in gbm.diag
  gbm.rank = data.table(gbm.diag[stat == "Median", .(mod, AUC, Log.Loss)])
  
  # rescale AUC and Log Loss to a [0, 1] range
  gbm.rank[, AUC := rescale(AUC, to = c(0, 1))]
  gbm.rank[, Log.Loss := 1 - rescale(Log.Loss, to = c(0, 1))]
  
  # compute the model score
  gbm.rank[, score := AUC + Log.Loss]
  
  # sort gbm.rank by score
  gbm.rank = gbm.rank[order(-score)]
  
  # define the model rank
  gbm.rank[, rank := 1:nrow(gbm.rank)]
  
  # just keep mod and rank in gbm.rank
  gbm.rank = gbm.rank[,.(mod, rank)]
  
} else if(Y.type == "Numeric")
{
  # extract the median Rsquared & RMSE of each model in gbm.diag
  gbm.rank = data.table(gbm.diag[stat == "Median", .(mod, Rsquared, RMSE)])
  
  # rescale Rsquared and RMSE to a [0, 1] range
  gbm.rank[, Rsquared := rescale(Rsquared, to = c(0, 1))]
  gbm.rank[, RMSE := 1 - rescale(RMSE, to = c(0, 1))]
  
  # compute the model score
  gbm.rank[, score := Rsquared + RMSE]
  
  # sort gbm.rank by score
  gbm.rank = gbm.rank[order(-score)]
  
  # define the model rank
  gbm.rank[, rank := 1:nrow(gbm.rank)]
  
  # just keep mod and rank in gbm.rank
  gbm.rank = gbm.rank[,.(mod, rank)]
  
} else
{
  # extract the median Micro.Accuracy & Multi.Log.Loss of each model in gbm.diag
  gbm.rank = data.table(gbm.diag[stat == "Median", .(mod, Micro.Accuracy, Multi.Log.Loss)])
  
  # rescale Micro.Accuracy and Multi.Log.Loss to a [0, 1] range
  gbm.rank[, Micro.Accuracy := rescale(Micro.Accuracy, to = c(0, 1))]
  gbm.rank[, Multi.Log.Loss := 1 - rescale(Multi.Log.Loss, to = c(0, 1))]
  
  # compute the model score
  gbm.rank[, score := Micro.Accuracy + Multi.Log.Loss]
  
  # sort gbm.rank by score
  gbm.rank = gbm.rank[order(-score)]
  
  # define the model rank
  gbm.rank[, rank := 1:nrow(gbm.rank)]
  
  # just keep mod and rank in gbm.rank
  gbm.rank = gbm.rank[,.(mod, rank)]
}

# make mod the key column in gbm.rank and gbm.diag
setkey(gbm.diag, mod)
setkey(gbm.rank, mod)

# join gbm.rank onto gbm.diag
gbm.diag = gbm.rank[gbm.diag]

# order gbm.diag by rank
gbm.diag = gbm.diag[order(rank)]

# export gbm.diag
write.csv(gbm.diag, "gbm-summary.csv", row.names = FALSE)

# add rank to gbm.cv
gbm.cv[, rank := gbm.rank$rank]

# add a Model identity column
gbm.cv[, Model := factor(ifelse(rank == 1, "Yes", "No"), levels = c("Yes", "No"))]

# plot our models performance metrics
# decide which type of model to run
if(Y.type == "Binary")
{
  gbm.plot = ggpairs(gbm.cv[,.(Accuracy, AUC, Log.Loss, Model)],
                     mapping = aes(color = Model, fill = Model),
                     upper = list(continuous = wrap("density", alpha = 1/4), combo = "box"),
                     lower = list(continuous = wrap("points"), combo = wrap("dot")),
                     diag = list(continuous = wrap("densityDiag", alpha = 1/3))) + 
    theme_bw(base_size = 20)
  
} else if(Y.type == "Numeric")
{
  gbm.plot = ggpairs(gbm.cv[,.(Rsquared, RMSE, MAPE, Model)],
                     mapping = aes(color = Model, fill = Model),
                     upper = list(continuous = wrap("density", alpha = 1/4), combo = "box"),
                     lower = list(continuous = wrap("points"), combo = wrap("dot")),
                     diag = list(continuous = wrap("densityDiag", alpha = 1/3))) + 
    theme_bw(base_size = 20)
  
} else
{
  gbm.plot = ggpairs(gbm.cv[,.(Macro.Accuracy, Micro.Accuracy, Multi.Log.Loss, Model)],
                     mapping = aes(color = Model, fill = Model),
                     upper = list(continuous = wrap("density", alpha = 1/4), combo = "box"),
                     lower = list(continuous = wrap("points"), combo = wrap("dot")),
                     diag = list(continuous = wrap("densityDiag", alpha = 1/3))) + 
    theme_bw(base_size = 20)
}

# set up a graphics window
# windows()

# set up a pdf file to capture the next graphic
pdf("gbm-plot.pdf", width = 16, height = 10, paper = "special") 

# call the plot
print(gbm.plot)

# close off the connection
dev.off()

# extract the model parameters of the best model
gbm.best = data.table(mod.id[mod == gbm.rank[rank == 1, mod]])

# predict the test set with the best model
# decide which type of model to run
if(Y.type == "Binary")
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Y, data = X,
                objective = objective, eval_metric = eval_metric,
                eta = gbm.best$eta, max_depth = gbm.best$max_depth, nthread = nthread,
                nrounds = gbm.best$nrounds, min_child_weight = gbm.best$min_child_weight,
                gamma = gbm.best$gamma, verbose = 0, scale.pos.weight = scale.pos.weight,
                subsample = gbm.best$subsample, colsample_bytree = gbm.best$colsample_bytree)
  
  # make predictions with the model on the test set
  ynew = as.numeric(as.numeric(predict(mod, newdata = as.matrix(test))) >= gbm.diag[mod == gbm.best$mod & stat == "Median", Cutoff])
  
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
  # build the training model
  set.seed(42)
  mod = xgboost(label = Y, data = X,
                objective = objective, eval_metric = eval_metric,
                eta = gbm.best$eta, max_depth = gbm.best$max_depth, nthread = nthread,
                nrounds = gbm.best$nrounds, min_child_weight = gbm.best$min_child_weight,
                gamma = gbm.best$gamma, verbose = 0,
                subsample = gbm.best$subsample, colsample_bytree = gbm.best$colsample_bytree)
  
  # make predictions with the model on the test set
  ynew = as.numeric(predict(mod, newdata = as.matrix(test)))
  
  # free memory
  gc()
  
  # add the predictions in ynew to test
  test[, Predictions := ynew]
  
} else
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Y, data = X, num_class = num_class,
                objective = objective, eval_metric = eval_metric,
                eta = gbm.best$eta, max_depth = gbm.best$max_depth, nthread = nthread,
                nrounds = gbm.best$nrounds, min_child_weight = gbm.best$min_child_weight,
                gamma = gbm.best$gamma, verbose = 0, weight = weight,
                subsample = gbm.best$subsample, colsample_bytree = gbm.best$colsample_bytree)
  
  # make predictions with the model on the test set
  ynew = predict(mod, newdata = as.matrix(test), reshape = TRUE)
  
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

# extract importance information for each variable
var.data = xgb.importance(model = mod, feature_names = colnames(X))

# determine if there are any variables that weren't even used
unused.vars = setdiff(colnames(X), var.data$Feature)

# if there were any unused variables then add them to gbm.imp with 0 values for each column
if(length(unused.vars) > 0)
{
  # make unused.vars into a data table
  unused.vars = data.table(Feature = unused.vars, 
                           Gain = rep(0, length(unused.vars)), 
                           Cover = rep(0, length(unused.vars)), 
                           Frequency = rep(0, length(unused.vars)))
  
  # append unused.vars to var.data
  var.data = rbind(var.data, unused.vars)
}

# rename the columns of var.data
setnames(var.data, c("Variable", names(var.data)[-1]))

# order var.data by Gain
var.data = var.data[order(Gain, decreasing = TRUE)]

# export test
write.csv(test, "gbm-test-predictions.csv", row.names = FALSE)

# export var.data
write.csv(var.data, "gbm-variable-data.csv", row.names = FALSE)

# free memory
gc()

}























