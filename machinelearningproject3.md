## Machine Learning Project 

# Background 


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. One set of 10 repetitions of the Unilateral Dumbbell Biceps Curl were undetaken in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

We wish to create a model that if we feed in someones exercise data, we'll be able to identify these common problems, or if the exercise was done correctly.  




```r
library(RCurl)
library(caret)
```



```r
# Read the training data for this project, ideally for reproducibility you'd
# just uncomment these 2 lines, but since I have slow internet and capped
# internet, I downloaded the file and initiate locally.

# x <-
# getURL('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
# training <- read.csv(text=x, stringAsFactors=F)

training <- read.csv("pml-training.csv")

# The testing data set for which we have to predict exercise groups:

x <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
testing <- read.csv(text = x)

# This is a messy dataset. Variables with many NA are not useful. I exclude
# these by combining all into 1 data set, extracting the useful variables,
# and then split again (mostly just to ease the modelling process later on):
names(testing) <- names(training)
data <- rbind(training, testing)
```

```
## Warning: invalid factor level, NA generated
```



```r
data[data == "#DIV/0!"] <- NA
data[data == ""] <- NA

temp <- rep(NA, 160)
for (i in 1:160) {
    # in an initial histogram I saw that the 0.2 is a convenient split to get
    # rid of NA dominated columns:
    if (mean(is.na(data[i])) <= 0.2) {
        temp[i] = names(data[i])
    }
}
temp <- na.omit(temp)  #this variable stores valid variable names
# remove non predictor values:
temp <- temp[-(1:7)]

## I now select the data
t2 <- data.frame(seq(1:nrow(data)))  #initialize dataframe

for (i in 1:160) {
    if (names(data[i]) %in% temp) {
        t2 <- cbind(t2, data[i])
    }
}
# remove initialization
t2 <- t2[-1]
sum(is.na(t2))  # this is a NA free dataset :)
```

```
## [1] 20
```

```r

# split back into initial datasets:
training <- t2[1:(nrow(t2) - 20), ]
testing <- t2[(nrow(t2) - 19):nrow(t2), ]
testing$classe <- seq(1:20)

# Set overall Seed

set.seed(111)

# Split for cross-validation
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
train <- training[inTrain, ]
test <- training[-inTrain, ]

## initial build of model using random forest

# THESE 3 LINES FOR PCA - WHICH MAKES A FASTER MODEL RUN preProc <-
# preProcess(train[, -53], method='pca', thresh=.95) pretrain <-
# predict(preProc, train[,-53]) pretrain <- cbind(pretrain, train[,53])
modFit2 <- train(classe ~ ., method = "rf", data = train)  #CHANGE DATA = PRETRAIN FOR PCA
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r

# here we can identify the key covariates and plot them:
modFit2$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.63%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3904    1    1    0    0    0.000512
## B   12 2639    7    0    0    0.007148
## C    0   21 2373    2    0    0.009599
## D    0    0   33 2218    1    0.015098
## E    0    0    1    8 2516    0.003564
```

```r

## Cross validation

# pretest <- predict(preProc, test[,-53]) # THESE 2 STEPS TO GO WITH PCA
# pretest <- cbind(pretest, test$classe)

predictions <- predict(modFit2, newdata = test)
confusionMatrix(predictions, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    2 1129    3    0    0
##          C    0    8 1022   24    0
##          D    0    0    1  939    4
##          E    0    0    0    1 1078
## 
## Overall Statistics
##                                        
##                Accuracy : 0.992        
##                  95% CI : (0.99, 0.994)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.99         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.991    0.996    0.974    0.996
## Specificity             1.000    0.999    0.993    0.999    1.000
## Pos Pred Value          0.999    0.996    0.970    0.995    0.999
## Neg Pred Value          1.000    0.998    0.999    0.995    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.174    0.160    0.183
## Detection Prevalence    0.284    0.193    0.179    0.160    0.183
## Balanced Accuracy       0.999    0.995    0.995    0.987    0.998
```

```r

## expected out of sample error is the 95% CI

# Now for the final answers: pretest <- predict(preProc, testing[,-53])
# pretest <- cbind(pretest, testing[,53])

answers <- predict(modFit2, newdata = testing)
```


## Why I made the choices I did  

The goal of this project was to predict the manner in which they did the exercise. There were many variables, most NA, and a simple na.omit(data) resulted in an empty dataset! So I applied a selective sorting to select on the principal data, ignoring many max, min, kurtosis etc elements derived from this data, and which must improve model performance for the official report (Velloso 2013). It would have been nice to select the 2.5s time window as this is suggested to improve model fit, but how to do this was not straightforward.  

In an initial rpart analysis, cross validation showed this model was not great 49.9 - 52.5%, so many of the 'error' movements (B - E) are mistakenly recorded as A (ie. correct way to do the exercise). So random forest trees are the way to go, even if you have to wait for your hours for the computer to process the model.  


# References

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

Read more: http://groupware.les.inf.puc-rio.br/har#dataset#ixzz37haeyycW

# Data   

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 
