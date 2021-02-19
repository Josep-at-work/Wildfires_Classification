######################################
## Model Building
## Evaluation of several classification algorithms
######################################

library(tidyverse)
library(magrittr)
library(lubridate)
library(dplyr)
library(ggplot2)
library(yardstick) #Tidy Characterizations of Model Performance
library(caret) #Classification And REgrassion Trainning
# detach("package:MASS", unload = TRUE)
wildfires <- read.csv('trabajo/wildfires.csv',row.names = 1 )

# First, reorder the features
detach("package:MASS", unload = TRUE) #necessary to avoid confusion with the select
wildfires %<>% select(c(X, Y, Square, Municipality, Day, Month, Year, Duration,FarmingArea, ForestalTotalArea, Cause))
wildfires$Cause <- as.factor(wildfires$Cause)
wildfires %<>% mutate(across(c(Square, Municipality), as.factor))
str(wildfires)  
##########################################################################################Intro####
# Based on the models from the book: ISL, I will compare the following classification models:
#   -Linear Discriminant Analysis
#   -Random Forest
#   -Boosting
#   -Extreme Gradient Boosting
# Then I will add an additional models:
#   -KNN  

######################################################################################Procedure####
# Using Nested CV is important when the sample size is small and when the choice of tuning 
# parameters is difficult. (For example, different seeds, different "best.tune"") 
# In Nested CV we separate parameter tuning of the models from choosing between the 
# models. The models also compete using the same folds of the data. 
# Thus, due to the size of this dataset, the best option is to use a Nested Cross-Validation
# approach to train and validate the models with the different combinations of hyperparameters
# Lastly use a test set to compare the best performing model of each algorithm. 


dim(wildfires) #678  13
# I take the most frequent category as the precision threshold to accept a model
mean('Arson' == wildfires$Cause) # 0.3938053

# In order to replicate my results I set a random seed.
set.seed(20012021)

# Train/Test Split: 
train_index = sample(nrow(wildfires), nrow(wildfires)*0.75)
train = wildfires[train_index,]
test = wildfires[-train_index,]

# Split train in 5 folds
ntrain = nrow(train)    
train.folds = createFolds(train$Cause,k=5,returnTrain=TRUE) #5 different boxes of the training data
# returnTrain=T returns 5 groups of 4/5 of the data to do training 
valid.fold = lapply(train.folds,function(x) (1:ntrain)[-x]) #
# Train Control adding one first layer of CV
fitControl <- trainControl(method = 'cv', number = 5, summaryFunction=defaultSummary)

##################################LDA####
## Linear Discriminative Analysis
######################################

# In the text book ISL with R the Logit for multi-class is not covered giving 
# more importance to the LDA for that matter then this will be the first model.

# I only used the numeric variables for the LDA, so I change the two factors for numbers.
# Also as there is some correlation I performed a PCA in order to remove variables collineality 
# and keep all information. 

lda.wildfires = wildfires %>% mutate(across(c(Square, Municipality), as.numeric))

lda.train = train %>% mutate(across(c(Square, Municipality), as.numeric))
                             # across(Cause, as.character))
lda.test = test %>% mutate(across(c(Square, Municipality), as.numeric))
                           # across(Cause, as.character))

varcor = cor(lda.wildfires[,-11])
which(abs(varcor)>0.75 & varcor!=1)

pca.train = prcomp(lda.train[,-11], scale=TRUE)

biplot(pca.train, scale=0, xlim=c(-1,3), ylim=c(-4,4))
pca.train$sdev -> pcsd

# Amount of variation each component captures
ggplot(data=as.data.frame(pcsd), aes(pcsd, x=1:10))+
  geom_line(color='green2') + geom_point() + 
  geom_hline(yintercept=1, linetype='dashed', color='red') + 
  ggtitle('Screen Plot',subtitle = "Variation captured by principal component") +
  xlab('PCs') + ylab('St. Dev.') + 
  theme_minimal() 

library(MASS) 

#LDA with PCA, LOOCV
set.seed(20012021)
lda.fit1 <- lda(Cause~pca.train$x[,1]+pca.train$x[,2]+pca.train$x[,3]+pca.train$x[,4],data=lda.train,CV=TRUE)
table(lda.fit1$class,train$Cause)
mean(lda.fit1$class==train$Cause) 
# Train Accuracy 0.4271654 

#Original variables, LOOCV
set.seed(20012021)
control <- trainControl(method='LOOCV')
lda.fit2 <- train(Cause~., data=lda.train,
                  method = 'lda', metric = 'Accuracy', trControl = control)
# Training Accuracy 0.4409449 

# Prediction with original variables
lda.pred <- predict(lda.fit2, newdata=lda.test)
lda.cm = confusionMatrix(lda.pred, lda.test$Cause)

# Performance
lda.cm                                    #confmatrix
lda.con$overall[1]                        #acc
# Test Accuracy = 0.4765

conf_mat(table(lda.pred, lda.test$Cause), type = "heatmap") -> cm
autoplot(cm,type='heatmap') +
  scale_fill_gradient2(low="firebrick1",mid = 'white', midpoint = 15, high = "chartreuse3") +
  theme_minimal() + 
  theme(legend.position = "right", axis.text.x = element_text(angle=45, vjust=1.1,hjust=1)) +
  ggtitle('Confusion Matrix LDA', subtitle = 'Test data')

detach('package::MASS', unload=T)
#############################RF####
## Random Forest
#################################

# Regarding Tree based models I'll jump directly to Random Forests and Boosting
# for better accuracy and lower variability.

library(randomForest)

# Simple model
set.seed(20012021)
rf.simple = randomForest(Cause~., data = train, mtry=as.integer(sqrt(10)), distribution='multinomial')
rf.simple #55.71% OOB-error (training error)

# By default the number of trees is 500, the chosen class for each observation is the most frequent of all the 
# trees in which an observation has been evaluated, generally in 500/3 trees,
# and the number of terminal nodes in each tree is 1.


# The training error can be estimated without performing CV which would be more computationally expensive.
# This is thanks to the OOB samples we have for each tree.

# Tuning 
getModelInfo()$rf$parameters
rfGrid = expand.grid(mtry=c(2,3,4,5,6,7,8,9,10))
rfacc <- 0
rftune = matrix(nrow = 5, ncol = 1) 
dimnames(rftune) = list(c("k1", "k2", "k3", "k4", "k5"), c("mtry")) 

# Simple CV
set.seed(20012021)
for (i in 1:5){ 
  
  train.data = train[train.folds[[i]],]
  validation.data = train[valid.fold[[i]],]
  
  rf.fit <- train(Cause~., data=train.data, method = 'rf',
                   tuneGrid = rfGrid, metric='Accuracy', distribution='multinomial')
  rf.fit
  
  rf.pred <- predict(rf.fit,validation.data) #uses best model to predict
  
  rfacc[i]=mean(rf.pred==validation.data$Cause)
  rftune[i,1] = as.matrix(rf.fit$bestTune)
}

rftune 
rfacc
mean(rfacc) #0.4805069
which.max(table(rftune)) -> position
mtry = rftune[position]
  
# Best model
set.seed(20012021)
rf.bst = randomForest(Cause~ . , data = train, mtry = 6, 
                 metric = 'Accuracy', distribution = 'multinomial')
rf.bst #OOB error 56.3%
oob.error.data = data.frame(
  Trees = rep(1:nrow(rf.bst$err.rate), times=9),
  Class = rep(c('OBB', 'Arson', 'Children', 'Engines', 'Lightning',
                'Miscellaneous', 'Smokers', 'Unknown', 'Vegetation Burning'),each = 500),
  Error = c(rf.best$err.rate[,1], rf.best$err.rate[,2], rf.best$err.rate[,3],
            rf.best$err.rate[,4], rf.best$err.rate[,5], rf.best$err.rate[,6],
            rf.best$err.rate[,7], rf.best$err.rate[,8], rf.best$err.rate[,9]))
ggplot(data=oob.error.data, aes(x=Trees, y=Error))+
  geom_line(aes(color=Class))+
  theme_minimal() + ggtitle('OOB Error',subtitle = "Classification error of the best model, m=6")
#From the error data we see that the error rate is stable at 500 trees so it's not necessary to add more trees.

names(rf.bst)
rf.bst$confusion 
heatmap(rf.bst$confusion[1:8,1:8], Rowv=NA, Colv=NA, scale='column') 
#This plot represents where the predictions of each group mainly fall in.

rf.pred <- predict(rf.bst, newdata = test)
rf.con = confusionMatrix(rf.pred, test$Cause)

# Performance
rf.con$table                             #confmatrix
rf.con$overall[1]                        #acc
# Test Accuracy = 0.4411765 

conf_mat(table(rf.pred, test$Cause), type = "heatmap") -> cm
autoplot(cm,type='heatmap') +
  scale_fill_gradient2(low="firebrick1",mid = 'white', midpoint = 15, high = "chartreuse3") +
  theme_minimal() + 
  theme(legend.position = "right", axis.text.x = element_text(angle=45, vjust=1.1,hjust=1))+
  ggtitle('Confusion Matrix RF', subtitle = 'Test data')

#From the results we see that the best performance was using 9 features to compete in each split.
#Also note that the bagging (m=p=10) has similar accuracy. The OOB error rate is 50% so half of the OOB samples are 
#correctly classified by the RF. And from the confusion matrix we see that 4 of the classes 0 true positives, with 
# an error rate of 1 that's probably due to the similarity between this categories; Engines, Miscellaneous, Smokers and Unkown.
table(wildfires$Cause)
#Eventhough the Miscellaneous category has more observations than the lightning one, it has a higher test-error rate
#that's probability because in this category there is a mix of distinct unusual categories so it's extremly difficult
#to find a pattern for this category, same with the Unknown one. The lack of accuracy predicting the fires
#originated by an engine or a smoker can be associated with the model performance or small amount of observation 
#available for this two categories.

#Importance of each variable, gini is a level of impurity
rf.importance = as.data.frame(rf.bst$importance)
rf.importance[order(importance),] #Square and municipality the most important

#####################################B#### 
## Boosting & XGBoost
#########################################

# The idea in boosting is to develop multiple weak prediction models sequentially,
# and that each of them uses the error of the previous ones in order to generate a stronger model,
# with better predictive power and better results stability.

library(gbm) 

set.seed(20012021)
getModelInfo()$gbm$parameters
#check tunable parameters
gbmGrid <-  expand.grid(interaction.depth = c(2,4,5,6,7,8,10), #Max Tree Depth 
                        n.trees = c(100,500,800),              #Boosting Iterations
                        shrinkage = c(.001, .005, .01, .05),   #learning rate
                        n.minobsinnode = c(5, 10, 20))          #Minimum Terminal Node Size

## Stochastic Gradient Boosting ##
gbmacc <- 0
gbmtune = matrix(nrow = 5, ncol = 4) #nrow=k, ncol=tuning parameters
dimnames(gbmtune) = list(c("k1", "k2", "k3", "k4", "k5"),  # row names 
            c("n.trees", "interaction.depth", "shrinkage", "n.minobsinnode")) # column names 
# Nested CV
for (i in 1:5){ 

  train.data = train[train.folds[[i]],]
  validation.data = train[valid.fold[[i]],]
  
  gbm.fit <- train(Cause~., data=train.data, method = 'gbm', trControl=fitControl,
                   tuneGrid=gbmGrid, metric='Accuracy', distribution='multinomial')
  gbm.fit
  
  gbm.pred <- predict(gbm.fit,validation.data) #uses best model to predict
  
  gbmacc[i]=mean(gbm.pred==validation.data$Cause)
  gbmtune[i,1:4] = as.matrix(gbm.fit$bestTune)
}

gbmtune 
gbmacc
mean(gbmacc) # 0.4883699

# Best boosting model
set.seed(20012021)
gbm.BST = gbm(Cause~.,data = train,n.trees = 500, #I use a larger n.tree and then select best iter for the prediction
        shrinkage = 0.005,interaction.depth = 7,
        cv.folds = 5, n.minobsinnode = 10,
        distribution = "multinomial")
# Check performance using 5-fold cross-validation
best.iter <- gbm.perf(gbm.BST,method="cv")
print(best.iter) # optimal number of trees
summary(gbm.BST,n.trees=best.iter)

# Predictions with test data
gbm.predBST <- predict(gbm.BST, n.trees=best.iter, newdata=test, type='response')
gbm.pred = as.matrix(gbm.predBST[,,1])
head(gbm.pred)
gbm.p.predBST = apply(gbm.predBST, 1, which.max)
gbm.p.predcat = colnames(gbm.predBST)[gbm.p.predBST] #giving category names
gbm.con = table(gbm.p.predcat,test$Cause)
gbm.test.acc = mean(gbm.p.predcat==test$Cause)

# Performance
gbm.con                            #confmatrix
gbm.test.acc                       #acc
# Test Accuracy = 0.4529412

conf_mat(table(gbm.p.predcat, test$Cause), type = "heatmap") -> cm
autoplot(cm,type='heatmap') +
  scale_fill_gradient2(low="firebrick1",mid = 'white', midpoint = 15, high = "chartreuse3") +
  theme_minimal() + 
  theme(legend.position = "right", axis.text.x = element_text(angle=45, vjust=1.1,hjust=1))+
  ggtitle('Confusion Matrix Boosting', subtitle = 'Test data')

#############################xgb####

library(xgboost)

str(wildfires)
# All variables must be numeric and the target must start with 0 for first class
str(train)

# One-Hot Encoding
order(table(wildfires$Municipality), decreasing = T) -> o_m
cumsum(prop.table(table(wildfires$Municipality)[o_m]))

# There are 48 municipalities, yet a 50% of the fires occurr in 8 of them. Then, each of this 
#  municipalities will have it's oun dummy features and a 9th group will be for the remaining. 
order(table(wildfires$Square), decreasing = T) -> o_s
cumsum(prop.table(table(wildfires$Square)[o_s]))
# Similarly with Square, 10 out of 52 values encompases a half of the observations.

dum.xgbdata = wildfires %>% mutate(across(c(Square, Municipality), as.character))
freq_mun = levels(wildfires$Municipality)[o_m][1:8]
freq_sq = levels(wildfires$Square)[o_s][1:10]
for (i in 1:nrow(dum.xgbdata)) {
  if (!(dum.xgbdata$Municipality[i] %in% freq_mun)){
    dum.xgbdata$Municipality[i] = 'Other_mun'
  }
  if (!(dum.xgbdata$Square[i] %in% freq_sq)){
    dum.xgbdata$Square[i] = 'Other_sq'
  }
}

dummy = dummyVars(~Municipality+Square, data=xgbdata)
new = predict(dummy, newdata=xgbdata)

dum.xgbdata %<>% cbind(new) %>% select(-c(Municipality, Square))

# Transformation needed for the xgboost() function, yet not for train() with xgb. With dummies.
dum.train.xgb <- dum.xgbdata[train_index,] %>% select(-c(Cause)) %>%
  as.matrix() %>% xgb.DMatrix(data=., label=(as.numeric(xgbdata[train_index,]$Cause)-1))
dum.test.xgb <- dum.xgbdata[-train_index,] %>% select(-c(Cause)) %>%
  as.matrix() %>% xgb.DMatrix(data=., label=(as.numeric(xgbdata[-train_index,]$Cause)-1))

# Transformation needed for the xgboost() function, yet not for train() with xgb. With no dummies.
xgbdata <- wildfires %>% mutate(across(c(Square, Municipality), as.numeric))

train.xgb <- train %>% select(-Cause) %>%
  mutate(across(c(Square, Municipality), as.numeric)) %>%
  as.matrix() %>% xgb.DMatrix(data=., label=(as.numeric(train$Cause)-1))

test.xgb <- test %>% select(-Cause) %>%
  mutate(across(c(Square, Municipality), as.numeric)) %>%
  as.matrix() %>% xgb.DMatrix(data=., label=(as.numeric(test$Cause)-1))

set.seed(20012021)
xgbmGrid <-  expand.grid(nrounds=c(100,200,500,700),
                         max_depth=c(2,4,6),
                         eta=c(0.001,0.005,0.01, 0.1),
                         gamma=c(0.01,0.001,0.0001), 
                         colsample_bytree=0.75,   #features suplyed to a tree
                         min_child_weight=0, 
                         subsample=1)   #all observations are given to each tree                               
# Dummy Nested CV
dum.xgbmacc <- 0
getModelInfo()$xgbTree$parameters
dum.xgbmtune = matrix(nrow = 5, ncol = 7) #nrow=k, ncol=tuning parameters
dimnames(dum.xgbmtune) = list(c("k1", "k2", "k3", "k4", "k5"),  # row names 
                         c("nrounds", "max_depth", "eta", "gamma","colsample_bytree" 
                           ,"min_child_weight", 'subsample')) # column names 
for (i in 1:5){ 
  
  train.data = dum.xgbdata[train_index,][train.folds[[i]],]
  validation.data = xgbdata[train_index,][valid.fold[[i]],]
  
  xgbm.fit <- train(Cause~., data=train.data, method = 'xgbTree', trControl=fitControl, 
                    tuneGrid=xgbmGrid, metric='Accuracy', distribution='multinomial')
  xgbm.fit
  
  xgbm.pred <- predict(xgbm.fit,validation.data)
  
  dum.xgbmacc[i]=mean(xgbm.pred==validation.data$Cause)
  dum.xgbmtune[i,1:7] = as.matrix(xgbm.fit$bestTune)
}

dum.xgbmtune
dum.xgbmacc
mean(dum.xgbmacc) #0.4645881

# Normal Nested CV
xgbmacc <- 0 
xgbmtune = matrix(nrow = 5, ncol = 7) 
dimnames(xgbmtune) = list(c("k1", "k2", "k3", "k4", "k5"), 
                          c("nrounds", "max_depth", "eta", "gamma","colsample_bytree" 
                            ,"min_child_weight", 'subsample')) 
for (i in 1:5){ 
  
  train.data = xgbdata[train_index,][train.folds[[i]],]
  validation.data = xgbdata[train_index,][valid.fold[[i]],]
  
  xgbm.fit <- train(Cause~., data=train.data, method = 'xgbTree', trControl=fitControl, 
                    tuneGrid=xgbmGrid, metric='Accuracy', distribution='multinomial')
  xgbm.fit
  
  xgbm.pred <- predict(xgbm.fit,validation.data)
  
  xgbmacc[i]=mean(xgbm.pred==validation.data$Cause)
  xgbmtune[i,1:7] = as.matrix(xgbm.fit$bestTune)
}

xgbmtune
xgbmacc
mean(xgbmacc) #0.4725473

# Best Model 
num_classes = length(unique(wildfires$Cause))
nrounds = 700 
set.seed(20012021)
xgb.bst <- xgboost(data = train.xgb, max.depth = 2, eta = 0.010, gamma = 0.0001, 
               nthread = 2, nrounds = nrounds, num_class = num_classes,
               min_child_weight = 0, subsample=1, 
               colsample_bytree=0.75,
               objective = "multi:softmax", 
               prediction = T, verbose = 2)

# Predict
bestxgb.pred <- predict(xgb.bst,  newdata = dum.test.xgb)
bestxgb.predcat = levels(wildfires$Cause)[bestxgb.pred + 1] #giving category names
xgb.con = table(bestxgb.predcat,test$Cause)
xgb.test.acc = mean(bestxgb.predcat==test$Cause)
bestxgb.predcat %<>% factor(levels=levels(wildfires$Cause))

# Performance
xgb.con                            #confmatrix
xgb.test.acc                       #acc
# Test Accuracy = 0.5294118

rf.bst$confusion
#Confusion matrix similar to the one with the random forest.
conf_mat(table(bestxgb.predcat, test$Cause), type = "heatmap") -> cm
autoplot(cm,type='heatmap') +
  scale_fill_gradient2(low="firebrick1",mid = 'white', midpoint = 15, high = "chartreuse3") +
  theme_minimal() + 
  theme(legend.position = "right", axis.text.x = element_text(angle=45, vjust=1.1,hjust=1)) +
  ggtitle('Confusion Matrix XGB', subtitle = 'Test data')

####################################KNN##### 
## K-Nearest Neighbors
#########################################

library(class)

#knn with a LOOCV

knn.data <- wildfires %>% select(-Cause) %>%
   mutate(across(c(Square, Municipality), as.numeric))
knn.train = knn.data[train_index,1:10]
y.train = as.numeric(wildfires$Cause)[train_index]
knn.test = knn.data[-train_index,1:10]
y.test = as.numeric(wildfires$Cause)[-train_index]
set.seed(20012021)
knn.acc = data.frame('k'=NA, 'Accuracy'=NA)
for (n in c(1:20)){
  knn.pred = knn.cv(knn.train, cl=y.train, k=n)
  a = mean(knn.pred==as.numeric(wildfires$Cause))
  knn.acc[n,1]=n
  knn.acc[n,2]=a
}
knn.acc %<>% na.omit()
ggplot(knn.acc, aes(x=k, y=Accuracy))+
  geom_line(color='red')+
  geom_point()+ 
  theme_minimal() + ggtitle('KNN',subtitle = "Accuracy using different k")
  

knn.acc[which.max(knn.acc$Accuracy),]
# Visually I select k = 9
knn.acc[which(knn.acc$k==9),] #0.3289086
knn.pred = knn.cv(knn.test, cl=y.test, k=9)
table(levels(test$Cause)[knn.pred],test$Cause)
mean(as.numeric(test$Cause)==knn.pred) #0.3823529

conf_mat(table(levels(test$Cause)[knn.pred], test$Cause), type = "heatmap") -> cm
autoplot(cm,type='heatmap') +
  scale_fill_gradient2(low="firebrick1",mid = 'white', midpoint = 15, high = "chartreuse3") +
  theme_minimal() + 
  theme(legend.position = "right", axis.text.x = element_text(angle=45, vjust=1.1,hjust=1))+
  ggtitle('Confusion Matrix KNN', subtitle = 'Test data')

####################################BstAlg##### 
## Best Algorithm, Gradient Boosting
######################################### 

set.seed(20012021)
finalGrid <-  expand.grid(interaction.depth = c(4,5,6,8), 
                        n.trees = c(100,500,800),              
                        shrinkage = c(.001, .005, .01),  
                        n.minobsinnode = c(5, 10, 20))
finalmodel <- train(Cause~., data=train, method = 'gbm', trControl=fitControl,
                    tuneGrid=gbmGrid, metric='Accuracy', distribution='multinomial')
finalmodel$bestTune
finalpred <- predict(finalmodel, test)
mean(finalpred==test$Cause) #0.5529412

########################CM####
# Confusion matrix, GBM on test data performed previously

confusionMatrix(as.factor(finalpred), test$Cause)$table %>%
  prop.table(margin = 1) %>%
  as.data.frame.matrix() %>%
  rownames_to_column(var = 'actual') %>%
  gather(key = 'prediction', value = 'freq',-actual) -> cm
cm$freq %<>% replace(cm$actual %in% c('Smokers', 'Unknown'), 0)
ggplot(cm, aes(x = actual, y = prediction, fill = freq)) +
  geom_tile() +
  geom_text(aes(label = round(freq, 2)), size = 3, color = 'gray20') + 
  scale_fill_gradient(low = 'yellow', high = 'red', limits = c(0,1), name = 'Relative Frequency') + 
  theme_minimal() + xlab('Prediction') + ylab('Truth')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1.1, size = 8)) +
  ggtitle('Confusion Matrix', subtitle =  'Boosting')

######################VI#### 
# Variable importance Boosting
final.imp = summary(finalmodel) #This variable importance comes from the training of the best GBoosting model done previously.
final.imp[which(final.imp$rel.inf!=0),]
final.imp$var %<>% replace(startsWith(final.imp$var, 'Mun'), 'Municipality') %>%
  replace(startsWith(final.imp$var, 'Sq'), 'Square')
aggregate(final.imp$rel.inf,by=list(final.imp$var), FUN=sum) %>%
  ggplot(aes(x=Group.1,y=x)) +
    geom_bar(stat='identity',aes(fill=Group.1))+
    theme_minimal()+ 
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank())+
    ylab('Relative Influence(%)') +
    ggtitle('Boosting Feature Importance', subtitle = 'Proportion of Gain in splits by feature') 

summary(finalmodel)

# Comparing the variable importance of Boosting with the XGB
importance <- xgb.importance(feature_names = NULL, model = xgb.bst)
importance
# Gain represents fractional contribution of each feature to the 
#   model based on the total gain of this feature's splits. Higher percentage means a more important predictive feature.
# Cover metric of the number of observation related to this feature
# Frequency percentage representing the relative number of times a feature have been used in trees
ggplot(importance, aes(x=Feature,y=Gain)) +
  geom_bar(stat='identity',aes(fill=Feature))+
  theme_minimal()+ 
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank())+
  ggtitle('XGB Feature Importance', subtitle = 'Proportion of Gain in splits by feature') 

ggplot(importance, aes(x=Feature,y=Frequency)) +
  geom_bar(stat='identity',aes(fill=Feature))+
  theme_minimal()+ 
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank())+
  ggtitle('XGB Feature Importance', subtitle = 'Relative number of times(%) a feature has been used in a tree') 
  
#############################Unknwon####
# Predict Unknown category with best algorithm 

train_set = wildfires %>% filter(Cause!='Unknown') %>% 
  droplevels('Unknown') 

test_set = wildfires %>% filter(Cause=='Unknown')
test_set$Cause %<>% factor(level='Unknown') 

set.seed(20012021)
Unknown.model <- train(Cause~., data=train_set, method = 'gbm', trControl=fitControl,
                    tuneGrid=gbmGrid, metric='Accuracy', distribution='multinomial')
# Predict
test_set %<>% filter(Municipality!='CONSELL')
Unknown.pred <- predict(Unknown.model, test_set)
Unknown.pred 

as.data.frame(table(Unknown.pred))[c(1,4,5,7),] -> data
colnames(data)=c('Category', 'Frequency')
ggplot(data, aes(x=Category,y=Frequency)) +
  geom_bar(stat='identity',aes(fill=Category))+
  theme_minimal()+ 
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank())+
  geom_text(aes(label = Frequency), size = 3, color = 'gray20')+
  ggtitle('Prediction of the Unknown', subtitle = 'Stochasting Gradient Boosting') 
