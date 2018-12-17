# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# set the path of where the input files are
mywd = "C:/ ... /Supervised/Variable-Selection"

# -----------------------------------------------------------------------------------
# ---- Packages ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# data handling
require(data.table)

# plotting
require(ggplot2)
require(scales)

# modeling
require(caret)
require(ranger)

# parallel processing
require(parallel)
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

}

# -----------------------------------------------------------------------------------
# ---- Set Up Data ------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# set the work directory
setwd(mywd)

# read in the data
train = data.table(read.csv("train.csv"))
info = data.table(read.csv("input-var-imp-supervised.csv", stringsAsFactors = FALSE))

# get the name of the response variable
Y.name = info[Metric == "response.variable.name", Value]

# get the data type of the response variable
Y.type = info[Metric == "response.variable.data.type", Value]

# get the proportion of cores that will be used for model building
cores = as.numeric(info[Metric == "cores", Value])

# remove any NA's in train and test
train = na.omit(train)

# update train to have just numeric predictor variables
train = cbind(unname(unlist(train[, Y.name, with = FALSE])), 
              data.table(model.matrix(~., data = train[, !Y.name, with = FALSE])[,-1]))

# update the column names of train
setnames(train, c(Y.name, names(train)[-1]))

# get predictors (X) and response (Y)
X = as.matrix(train[,!Y.name, with = FALSE])
Y = as.character(unname(unlist(train[, Y.name, with = FALSE])))

# create a copy of Y
if(Y.type != "Numeric")
{
  Y.copy = as.factor(Y)
  
} else
{
  Y.copy = as.numeric(Y)
}

# set up model parameters
if(Y.type == "Binary")
{
  # create a table to map the old levels of Y into the new levels
  Y.mapping = data.table(old.level = levels(Y.copy),
                         new.level = 0:(length(levels(Y.copy)) - 1))
  
  # reset the names of the levels of Y
  Y = as.numeric(Y.copy) - 1
  
  # compute the number of instances across the 2 classes
  cases = table(Y)
  
  # determine the number of negative and positive cases
  cases = data.table(negative = cases[which(names(cases) == 0)],
                     positive = cases[which(names(cases) == 1)])
  
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
  
  # make Y into a factor
  Y = factor(Y, levels = 0:1)
  
} else if(Y.type == "Numeric")
{
  # update Y
  Y = Y.copy
	
	# compute the number of classes
  num_class = 0
    
} else
{
  # create a table to map the old levels of Y into the new levels
  Y.mapping = data.table(old.level = levels(Y.copy),
                         new.level = 0:(length(levels(Y.copy)) - 1))
  
  # reset the names of the levels of Y
  Y = as.numeric(Y.copy) - 1
  
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
  
  # make Y into a factor
  Y = factor(Y, levels = 0:(num_class - 1))
}

# remove objects we no longer need
rm(info, Y.copy, mywd)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Variable Importance ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Correlation ------------------------------------------------------------------

{

# extract all variables to test correlation
cor.dat = data.table(X)

# set up the data for modeling
mod.dat = data.table(cbind(y = Y, cor.dat))

# compute correlations
cors = cor(cor.dat)

# replace any NA's with 0's
cors[is.na(cors)] = 0

# find out which variables are highly correlated (by magnitude) and remove them
find.dat = findCorrelation(cors, cutoff = 0.9, names = TRUE, exact = TRUE)

# remove columns from mod.dat according to find.dat
if(length(find.dat) > 0) mod.dat = mod.dat[, !find.dat, with = FALSE]

}

# ---- Importance -------------------------------------------------------------------

{

# choose how many threads to use
nthread = max(1, floor(cores * detectCores()))

# decide which type of model to run
if(Y.type == "Numeric")
{
  # build the training model
  set.seed(42)
  mod = ranger(y ~ ., 
               data = mod.dat,
               num.trees = 500,
               min.node.size = 1,
               num.threads = nthread,
               importance = "impurity",
               seed = 42)
  
  # free memory
  gc()
  
} else
{
  # build the training model
  set.seed(42)
  mod = ranger(y ~ ., 
               data = mod.dat,
               num.trees = 500,
               min.node.size = 1,
               case.weights = weight,
               num.threads = nthread,
               importance = "impurity",
               seed = 42)
  
  # free memory
  gc()
  
}

# extract variable importance
var.imp = importance(mod)

# make var.imp into a table
var.imp = data.table(variable = names(var.imp), 
                     value = as.numeric(var.imp))

# put importance on a 0-1 scale for easy comparison
var.imp[, value := rescale(value, to = c(0, 1))]

# order by importance
var.imp = var.imp[order(value, decreasing = TRUE)]

# make variable a factor for plotting purposes
var.imp[, variable := factor(variable, levels = unique(variable))]

# plot a barplot of variable importance
var.imp.plot = ggplot(var.imp, aes(x = variable, y = value, fill = value, color = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Variable Importance\nRandom Forests") +
  labs(x = "Variable", y = "Scaled Importance") +
  scale_y_continuous(labels = percent) +
  scale_fill_gradient(low = "yellow", high = "red") +
  scale_color_gradient(low = "yellow", high = "red") +
  theme_dark(30) +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5), axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major.x = element_blank())

# set up a graphics window
# windows()

# set up a pdf file to capture the next graphic
pdf("var-imp-plot.pdf", width = 16, height = 10, paper = "special") 

# call the plot
print(var.imp.plot)

# close off the connection
dev.off()

# add a rank column to var.imp
var.imp[, rank := 1:nrow(var.imp)]

# export var.imp
write.csv(var.imp, "var-imp-rank.csv", row.names = FALSE)

# free memory
gc()

}

}



