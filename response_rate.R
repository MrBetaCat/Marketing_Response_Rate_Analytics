library(data.table)
library(xgboost)
set.seed(66)
#use fread to accelerate the data input speed 
#fread is much faster than read.csv()
train=fread('train.csv', header = T, sep = ',', stringsAsFactors=TRUE) 
#convert to data.frame
train=as.data.frame(train)
test=fread('test.csv', header = T, sep = ',', stringsAsFactors=TRUE) 
#convert to data.frame
test=as.data.frame(test)

###data engineering part
#convert factor features to numeric features if we wanna use xgboost
indx <- sapply(train, is.factor)
train[indx] <- lapply(train[indx], function(x) as.numeric(factor(x)))
test[indx] <- lapply(test[indx], function(x) as.numeric(factor(x)))

###dealing with missing values
#check the missing value summary, this method is not very efficient in R
#I guess the main reason is that "is.na" is not efficient 
table(is.na(train))
table(is.na(test))

###replacing missing values with -1
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

###split training data into training part and validate part
## 70% of the sample size
smp_size <- floor(0.70 * nrow(train))
## set the seed to make your partition reproductible
train_ind <- sample(seq_len(nrow(train)), size = smp_size)
dtrain <- xgb.DMatrix(data = data.matrix(train[train_ind,2:ncol(train)-1]),label = train$target[train_ind])
dval <- xgb.DMatrix(data = data.matrix(train[-train_ind,2:ncol(train)-1]),label = train$target[-train_ind])

## set parameters for xgboost
watchlist <- list(eval = dval, train = dtrain)
param <- list(  objective           = "binary:logistic", 
                eta                 = 0.020,
                max_depth           = 6, 
                eval_metric         = "auc"
)

## training a XGBoost classifier 
clf <- xgb.train(   params              = param,
                    data                = dtrain, 
                    nrounds             = 200, 
                    verbose             = 1, 
                    early_stop_round    = 20,
                    watchlist           = watchlist,
                    maximize            = TRUE)
#running results:
#eval-auc:0.772939	train-auc:0.814457

##The following code is for final testing data prediction
# Making predictions in batches if the computer has memory limitation
predict_res <- data.frame(ID=test$ID)
predict_res$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
  predict_res[rows, "target"] <- predict(clf, data.matrix(test[rows, 2:ncol(train)-1]))
}
#Otherwise
#predict_res <- predict(clf, data.matrix(test[rows, 2:ncol(train)-1]))

## Output the final testing data prediction results into csv file
write_csv(predict_res, "response_rate_test_res.csv")
