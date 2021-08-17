# setwd('C:/Users/iliu2/Documents/Downloads/afl')
rm(list=ls())
setwd('/Users/ivanliu/Google Drive/Competition/Sportsbet_AFL_Competition')
# 0. Read the Data
teams <- read.csv("data/teams.csv", stringsAsFactors = F)
seasons <- read.csv("data/seasons.csv", stringsAsFactors = F)
unplayed <- read.csv("data/unplayed.csv", stringsAsFactors = F)

# 1. Generate Training Data
# Derive training data from the 2011 to 2014 AFL season statistics
train_df <- seasons[which(seasons$season %in% c(2011:2014)),]
# There were drawn games during this period
train_df[which(train_df$margin == 0),]
# For simplicity, we will omit them from our training data.
train_df <- train_df[which(train_df$margin > 0),]
# Generate the "gold-standard" tid1 win probabilities for each match in df_train as a new column called prob
train_df$prob <- ifelse(train_df$tid1 == train_df$win_tid, 1,0)
# Extract features
feat_cols <- c("mid", "prob", "round", "tid1", "tid2", "tid1_loc")
train_df <- train_df[,feat_cols]
# Co-erce the round values from strings to integers (for later use as an explicit ordinal feature):
train_df$round <- as.integer(substr(train_df$round,2,length(train_df$round)))
# The remaining features (tid1, tid2, tid1_loc) are categorical features that need to be numerically-encoded using binary dummy variables:
train_df$tid1 <- as.factor(train_df$tid1)
train_df$tid2 <- as.factor(train_df$tid2)
train_df$tid1_loc <- as.factor(train_df$tid1_loc)
#train_df$prob <- as.factor(train_df$prob);#levels(train_df$prob) <- c('Yes','No')
train_df$prob <- ifelse(train_df$prob == 1, 'Yes', 'No')

# Caret Model
library(caret)
# y: prob | x: -prob, -mid
trainIdx <- createDataPartition(train_df$prob, p = .7,list = FALSE)
train <- train_df[trainIdx,-1]
test <- train_df[-trainIdx,]
#dummies <- dummyVars(~prob, data = train[,c('tid1','tid2','tid1_loc','prob')])
#y <- predict(dummies, newdata = train[,c('tid1','tid2','tid1_loc','prob')])

fitControl <- trainControl(method = "adaptive_cv",
                           number = 10,
                           repeats = 5,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary,
                           ## Adaptive resampling information:
                           adaptive = list(min = 10,
                                           alpha = 0.05,
                                           method = "gls",
                                           complete = TRUE))

set.seed(825)
fit <- train(y=train$prob, x=train[,-1], data=train,
                method = "rf",
                trControl = fitControl,
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "ROC")

# Test
p <- predict(fit, test, type='prob')

LogLoss<-function(actual, predicted)
{
  predicted<-(pmax(predicted, 0.00001))
  predicted<-(pmin(predicted, 0.99999))
  result<- -1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

LogLoss(as.numeric(test$prob)-1,as.numeric(p)-1)



