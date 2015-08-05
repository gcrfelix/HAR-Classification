library(caret)

#################################################
# data prep
#################################################

# load data
temp <- tempfile()
download.file("http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip", temp)
har <- read.csv(unz(temp, "dataset-har-PUC-Rio-ugulino.csv"), head=TRUE, sep = ";")
unlink(temp)

# change attribute name
names(har)[4] <- "height"
names(har)[6] <- "bmi"

#function to change comma to dot 
myfun <- function(x) {sub(",",".",x)} 

#apply the function to "height", "bmi" and "z4" variables, and convert each variable to num type
var1 <- as.numeric(sapply(har$height, FUN=myfun)) 
har$height <- var1 

var2 <- as.numeric(sapply(har$bmi, FUN=myfun)) 
har$bmi <- var2

var3 <- as.numeric(sapply(har$z4, FUN=myfun)) 
har$z4 <- var3

# omit missing or NA data
har <- na.omit(har)

# prune 'name' attribute which is of no use for classification
new_har <- har[,-1]

# dummy variables for factors/characters
dataDummy <- dummyVars("class~.",data=new_har, fullRank=T)
new_har <- as.data.frame(predict(dataDummy,new_har))
new_har[,"class"] <- har[,"class"]

#################################################
# modeling
#################################################

# get names of all caret supported models 
names(getModelInfo())

# split data into training and testing chunks
set.seed(1234)
splitIndex <- createDataPartition(new_har$class, p = .9, list = FALSE)
training <- new_har[splitIndex,]
testing <- new_har[-splitIndex,]

# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv',number=10)

# Models fitting on train set > 5 minutes have been discared.
library(doParallel)
registerDoParallel()

# create an table to compare different models
table <- data.frame(model=character(), time=numeric(), accuracy=numeric(), stringsAsFactors=FALSE)

# model 1: conditional inference tree
startTime <- as.integer(Sys.time())
ctreeFit <- train(training, training$class, "ctree", tuneLength = 5, trControl=objControl)
endTime <- as.integer(Sys.time())
duration <- (endTime-startTime)/60
table[1,1] <- "ctree"
table[1,2] <- duration
ctreeFit
plot(ctreeFit$finalModel)

# model 2: C5.0 Decision Tree
startTime <- as.integer(Sys.time())
C50Fit <- train(training, training$class, "C5.0", tuneLength = 5, trControl=objControl)
endTime <- as.integer(Sys.time())
duration <- (endTime-startTime)/60
table[2,1] <- "C5.0"
table[2,2] <- duration
C50Fit
C50Fit$finalModel

# model 3: CART
startTime <- as.integer(Sys.time())
rpartFit <- train(training, training$class, "rpart", tuneLength = 5, trControl=objControl)
endTime <- as.integer(Sys.time())
duration <- (endTime-startTime)/60
table[3,1] <- "CART"
table[3,2] <- duration
rpartFit
rpartFit$finalModel

#################################################
# model evaluation and comparision
#################################################

ctree_results <- predict(ctreeFit, testing)
table[1,3] <- postResample(pred=ctree_results, obs=as.factor(testing[,"class"]))[1]

c50_results <- predict(C50Fit, testing)
table[2,3] <- postResample(pred=c50_results, obs=as.factor(testing[,"class"]))[1]

rpart_results <- predict(rpartFit, testing)
table[3,3] <- postResample(pred=c50_results, obs=as.factor(testing[,"class"]))[1]

# compare models
resamps <- resamples(list(ctree=ctreeFit, C50=C50Fit, CART=rpartFit))
resamps
summary(resamps)

print(table)

#################################################
# conclusion
#################################################
# from the table, we can see that all the three classification methods achieve 100% accuracy 
# while the fastest model in fitting training data is CART. So CART should be the best classifier.
