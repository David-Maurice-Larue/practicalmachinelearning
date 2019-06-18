---
title: "Predicting class of weightlifting from biometric data"
author: "David Larue"
date: "June 17, 2019"
output: 
  html_document: 
    keep_md: yes
---



## Summary

With data from the *WLE dataset* of Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013, we train a Machine Learning algorithm to predict which type or error or lack thereof several weightlifters engaged in based on biometric information gathered. We employ a Random Forest to train, validate and finally test, with good success.

## Body

### Preliminaries

Data is downloaded and loaded, and R libraries loaded.


```r
# setwd("C:/Users/david/Documents/R/datasciencecoursera/datascience-course8/project")
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")
# pml.testing<-read.csv("pml-testing.csv",na.strings=c("", "NA"))
# pml.training<-read.csv("pml-training.csv",na.strings=c("", "NA"))
# save(list=c("pml.testing","pml.training"),file="pml.RData")
load(file="pml.RData")
library(dplyr)
library(caret)
library(randomForest)
```

### Data Exploration and Cleaning

Note that "classe" is the last (160th) variable in pml.training, which is replaced by problem.id in pml.testing. This tells us that we must do our training and validation on subsets of pml.training, and that pml.testing is the set of predictor values for the final quiz, and that the predicted value will be a letter from "A" to "E".

First we kill off all variables that have NA or blank values. We shall not be using the timestamps (raw_timestamp_part_[12],cvdt_timestamp) or new_window (as all "no") or "X" (no info).  This reduces the number of variables from 160 to 55. There are 19,622 observations.


```r
num.nas<-function (var.name) {sum(is.na(pml.training[,var.name]))}
no.nas.index<-mapply(num.nas,names(pml.training))==0
pml.testing<-pml.testing[,no.nas.index];pml.training<-pml.training[,no.nas.index]
col.omit=c("X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window")
pml.testing<-select(pml.testing,-col.omit);pml.training<-select(pml.training,-col.omit)
```

We split the provided data into training and validation sets.


```r
set.seed(2)
inTrain <- createDataPartition(y=pml.training$classe, p=0.7, list=FALSE)
training <- pml.training[inTrain,]
validation <- pml.training[-inTrain,]
```

### Description of how the model was built

#### Why the choices were made

A Random Forest algorithm (*rf*) from the caret package was chosen, as it has a reputation for good predictive success. As it turns out, it was completely successful, and no other models or parameters were investigated.

The choice of 100 for ntree was made to minimize time spent, and because experience shows that things stabilize well before 1,000. 

Because many of the parameters are factor-like (integers with a small range of values), a small value for mtry, 3, was specified to save time and to yet include enough samples to end up being accurate.

If it were possible to use parallel processing, this was explicitly permitted,

The *classe* variable was the outcome, from 'A' through 'E'. The model was built against all the remaining variables, all of which came into the data set as numeric.

### How cross validation was used

It was requested to include some Cross Validation, and this was chosen with a small parameter of 2. The caret package handles the details.


```r
set.seed(3)  
system.time({modFit <- train(classe~ .,data=training,  
                             method="rf",  
                             trControl=trainControl(method="cv",number=2), 
                             ntree = 100,  
                             tuneGrid = data.frame(mtry = 3),  
                             prox=TRUE,  
                             allowParallel=TRUE)})
```

```
##    user  system elapsed 
##  103.41    2.06  112.71
```

A modest amount of time, 100 seconds or so, was consumed to build this model.

We first confirm that the model (*modFit*) performs adequately on the training set itself.


```r
right.tr<-sum(predict.train(modFit,training)==training$classe)
total.tr<-length(training$classe)
proportion.tr<-right.tr/total.tr
c(right.tr, total.tr,proportion.tr)
```

```
## [1] 13737 13737     1
```

Perhaps surprisingly, this model gets 100% of the 13,737 outcomes in the training partition correct. This raises the specter of whether over-fitting will be a problem. We now check the performance on the remaining 5,885 observations in the validation partition.


```r
right.v<-sum(predict.train(modFit,validation)==validation$classe)
total.v<-length(validation$classe)
proportion.v<-right.v/total.v
c(right.v, total.v,proportion.v)
```

```
## [1] 5864.0000000 5885.0000000    0.9964316
```

At 99.64% (5864/5885), we are satisfied with the performance here.  "Good enough for government work." 

### What might be expected for the out of sample error

The error rate of 0.00% on the sample that the model was trained on rose slightly, as expected, to 0.36% when predictions of the virginal validation set were made. This is the estimate for the out of sample error when the model is used to predict on any sample.

We have a final test sample to apply this to, of 20 observations. As 0.36% of 20 = 0.072 << 1, we have some confidence that these predictions will be all correct.


```r
testing.result<-predict.train(modFit,pml.testing)
t(testing.result)
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
## [1,] B    A    B    A    A    E    D    B    A    A     B     C     B    
##      [,14] [,15] [,16] [,17] [,18] [,19] [,20]
## [1,] A     E     E     A     B     B     B    
## Levels: A B C D E
```

And indeed the submission was graded as 100% successful, 20/20.

## Conclusion

The Random Forest algorithm applied to this data set performed well, and made predictions with high accuracy.
