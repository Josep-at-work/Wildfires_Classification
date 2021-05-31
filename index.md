---
title: "Wildfires Classification"
output:
  pdf_document: 
    fig_caption: true
    latex_engine: xelatex
    includes:
      in_header: ImagePositionH.txt
  pdf: default
---

The motivation for this project comes from the increase in the amount of wildfires occurring worldwide over time, which not only provokes the direct damage to inhabitants, the native flora and fauna, but also has an indirect effect on the climate change. From one side, the combustion of forests release $CO_2$ and other greenhouse gasses to the atmosphere. Moreover, a greater impact on the atmosphere is caused when the dead plants decompose over the following decades, plus the amount of $CO_2$ that these trees will not be removing from the atmosphere as living trees would. 

With all the data collection happening currently, I believe that some interesting topics of research could be done in this area. An obvious one is the prediction of the probability of a wildfire happening as it would allow the firefighters to be ready on the right time and the right place with higher accuracy. Also predicting the cause once the catastrophe has taken place can be of help as many fires remain uncertain about its original cause.  

# 1. About the data        

```{r include=FALSE}
knitr::opts_knit$set(root.dir="~/MADM/Aprenentatge Estadístic/Trabajo")

library(tidyverse)
library(dplyr)
library(magrittr)

library(kableExtra)

wildfires <- as.data.frame(read.csv('Incendios2.csv'))
```

The data set I've used for this project contains a detailed description of all wildfires occurred in Mallorca from 2010 to 2019. Each observation has many features such as the origin's coordinates and timestamp, as well as number of affected municipalities, burnt area and cause for instance. 

# 2. Data Wrangling
       
## 2.1 Feature Selection
\   
First thing I did was to take a glimpse to all features to select the most relevant and usable ones. Some variables where directly deleted for having one unique value on the whole data set, probably because this data is part of a larger set. Other features were removed due to it's lack of information such as the number of municipalities involved on each wildfire.

```{r echo=FALSE}
as.data.frame(table(wildfires$NumeroMunicipiosAfectados)) -> num.mun
colnames(num.mun) = c("Number of Municipalities", "Frequency")
kable(num.mun, align = "ll", caption = "Number of Municipalities affected by the fires", booktabs = T) %>%
  kable_styling(latex_options = "hold_position", font_size = 7)
```
Other variables were just necessary for the administration to identify the fires, yet not for my project. 

```{r echo=FALSE}
a=data.frame('Features'=c('Year','Municipality','Square','X','Y', "Detected", "Extinted", "Cause", "Motivation", "FarmingArea", "NoTreesArea", "ForestalTotalArea"),
             'Description'=c('Year','Municipality','Origin\'s location on a map',                     "Coordenate X","Coordenate Y", "Origin timestamp", "Extintion timestamp", "Fire Cause", "Fire Motivation", "Farming area burnt (ha)", "Area without trees burnt (ha)", "Forest total area burnt (ha)"))
kable(a, align = "ll" ,caption = "Preselected Features", booktabs = T) %>%
  kable_styling(latex_options = "hold_position", font_size = 7) 
```


Next, I gave the corresponding data type to each feature:  

+ **Number**: *FarmingArea*, *NoTreesArea*, *ForestalTotalArea*, *X*, *Y*, *Year*.  
+ **Datetime** ('%d/%b/%Y %H:%M:%S'): *Detected*, *Extinted*.   
+ **Character**: all others

## 2.2 Feature Engineering  
\ 
In this part of the project I proceeded to modify the *datetime* features. I thought that a more suited variable for the learning of the algorithms would be to compress *Detected* and *Extinted* variables into a variable containing the total time, in minutes, that the fire was alive. Also from the *Detected* values I extracted the day of the week and month of the year in which the fire was originated. In this process I removed two features and added three new ones.  


![DurationVsArea](Images/DurationVsArea.png){align=center}

## 2.3 Target Variable  
\  
Initially there were 46 different labels describing the cause of the fires, which off course would be overwhelming for any algorithm to classify. So I proceeded to analyze the frequency of the values. 
First, I removed 10 observations which cause was classified just once, also I confirmed that the total area burnt by these causes was minimum.   
Then, after and exhaustive analysis of all different causes I compressed the remaining ones in 8 distinct causes which I found they are the main groups in wildfires classification. 

```{r echo=FALSE, fig.align='center', fig.pos='h', out.height="50%", out.width="75%", fig.cap='Number of observations by Cause'}
knitr::include_graphics('Images/CountByCause.png')
```

The causes were reduced from 46 to 8. 
In addition, during this analysis I found out that the *Motivation* variable is only a short description added to the *arson* wildfires, so I dropped this feature.  

# 3. EDA  

### Categorical Variable  
\    
Some algorithms can not deal with categorical variables, yet due to the large number of different municipalities in this set a *OHE* wouldn't be convenient. During the model development I used to diferent procedures in order to fit the models with just numeric variables. This process was applyed to the *XGBoost* model for instance.  

![Number of Wildfiresfires by Municipality](Images/FiresByMunicipality.png){align=center}

### Distribution of fires in time.  
\  
```{r echo=FALSE, fig.align='center', fig.pos='h', out.height="40%", out.width="75%", fig.cap='Area of forest burnt by day of the week'}
knitr::include_graphics('Images/AreaBurntByDay.png')
```

![Area of forest burnt by month](Images/AreaBurntByMonth.png){align=center}

### Feature Correlation   
\  
For the correlation analysis I removed the categorical variables.  

![Correlation with significancy](Images/Correlation.png){align=center}

The total area of forest burnt and the area of forest without trees burnt are almost perfectly correlated with statistical significance, thus I removed the *NoTreesArea*. Farming area burnt is correlated with the total area of forest too, yet I kept it for the model development.

# 4. Model Development  
\     
This section presents the analysis of different algorithms' performance on classifying the wildfires in 8 distinc categories. The algorithms I decided to compare are the followings:

  + Linear Discriminant Analysis   
  + Random Forest   
  + Boosting   
  + XGBoosting    
  + KNN  
   
This selection is based on the multiclass classification algorithms introduced in the text book *Introduction to Statistical Learning*. Regarding the first four algorithms, they all learn in a supervised way looking for patterns in the training data and are listed in descending order of complexity. While the KNN is also a supervised learning algorithm, it uses a proximity measure method in order to classify the observations. Due to the problem's nature, the **accuracy** will be measured in the **proportion of true positives** classified by each model.  

First, I loaded the final data, modified the feature's order and transformed the character variables to factor, which is demanded by many algorithms in *R*. 
```{r echo=FALSE, result="asis"}
wildfires <- read.csv('wildfires.csv',row.names = 1)
wildfires %<>% select(c(X, Y, Square, Municipality, Day, Month, Year, Duration,FarmingArea, ForestalTotalArea, Cause))
wildfires %<>% mutate(across(c(Square, Municipality, Cause), as.factor))
str(wildfires)
```

Then I divided the data in training and test sets in a **75-25% random split**.
```{r echo=FALSE}
set.seed(20012021)
train_index = sample(nrow(wildfires), nrow(wildfires)*0.75)
train = wildfires[train_index,]
test = wildfires[-train_index,]
kable(head(train[-11]), caption = 'Training set, explanatory variables', booktabs = T) %>% 
  kable_styling(latex_options = c('hold_position','scale_down'), font_size = 6)
kable(head(train[11]), caption = 'Training set, target', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

As you can see from the tables above, the set in use is relatively small, thus I decided to use a **Nested CV** approach for the hyperparameter tuning of the *boosting* and *XGB* algorithms. Hence, I randomly separated the training data again in 5 equal-sized folds and saved it for later on. 

The followed procedure to find the best model was to train all algorithms where the best training performance resulted with the *Boosting* with a $48.84\%$ of train accuracy. However, I decided to compute the test accuracy of the 5 other models too, so that I could better compare their performances with a confusion matrix of their classification on the test data. In the last section of model developing I did a grid search to find the optimal parameters for the *boosting* algorithm on the training data and finally compute the test accuracy of the best model. 

## 1. LDA   
\   
*LDA* requires all variables to be numeric, thus I transformed the factors to numbers. Next, in order to avoid problems of collineality between variables and keep maximum information possible, I computed the principal components of my features to fit the *LDA* model. 
```{r echo=FALSE, fig.align='center', fig.pos='h', out.height="35%", out.width="75%", fig.cap='Principal Component\'s st.dev'}
knitr::include_graphics('Images/pcsd.png')
```

I kept the first four principal components as they have an *st.dev* over 1. Then, at this point I trained two models, one using the four first principal components and a second one using the original variables. Both training accuracy was quite similar, slightly better on the second model. Hence, I used the second model to predict the test data. 

$$
Cause(PC1,PC2,PC3,PC4) \ (M1) \\
$$
$$
Cause(X,Y,Square,Municipality,Day,Month,Year,Duration,FarmingArea,ForestalTotalArea) \ (M2)
$$

```{r echo=FALSE}
Models = c('M1', 'M2'); Training_Acc = c(0.4271654,0.4409449); Test_Acc = c(NA, 0.4765)
lda.results = data.frame(Models, Training_Acc, Test_Acc)
kable(lda.results, caption = 'LDA Performance', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

![Confusion Matrix LDA](Images/cmlda.png){align=center, heigth=100%, weigth=100%}

## 2. Random Forest    
\      
Regarding the Tree based models, I jumped directly to *Random Forest* and *Boosting* for **better accuracy** and **lower variability**.  

By default the number of trees is 500 and the predicted class for each observation is the most frequent of all resulting values from the trees in which an observation has been evaluated, generally in `r round(500 * 2/3,2)` out of the 500 trees, i.e. $2/3$. This means that each tree is not developed using the whole data set, the other third of the observations which are not used to develop an specific tree, namely the **Out of Bag** sample, is then used to evaluate the model. Thus, the validation error can be computed without using a CV approach which would be more computationally expensive. 

This algorithm has a tuning parameter *m*, which is the number possible features to use in each split. If the *m* is smaller than the total number of variables, the algorithm randomly selects *m* variables to compete for the next split. 

After performing a hyperparameter searching, the optimal value for *m* happened to be 6 in two of the 5 iterations and the following plot shows it's training performance:

```{r echo=FALSE, fig.align='center', fig.pos='h', out.height="75%", out.width="75%", fig.cap='Out of Bag Error'}
knitr::include_graphics('Images/OOBError.png')
```


It seems clear that the error rate is stable at 500 trees so it is not necessary to develope a larger forest.  

Finally, after training the optimal model and testing with the respective sets, the model's performance was:

```{r echo=FALSE}
Models = c('m=5'); Training_Acc = c(0.4805069); Test_Acc = c(0.4411765)
rf.results = data.frame(Models, Training_Acc, Test_Acc)
kable(rf.results, caption = 'Random Forest Performance', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```
For this algorithm the $TrainingAcc= \frac{100-OBB_{Error}}{100}$, where the $OBB_{Error}$ is in %.

![Confusion Matrix Random Forest](Images/cmrf.png){align=center, heigth=100%, weigth=100%}

## 3. Boosting  
\      
The idea in boosting is to develop multiple weak prediction models sequentially, in a way that each of them uses the error of the previously developed ones in order to generate a stronger model, with better predictive power and better results stability. 

Applying boosting to decision trees rises the complexity of the algorithm. In this case there are four important parameters to take into account: the **maximum depth of the each individual tree**, the **number of boosting iteration**, i.e trees, the **learning rate** and the **minimum size of each leaf**. Hence, again I performed a hyperparameter searching to find the optimum combination of parameters for my problem. Now, the boosting algorithm do not uses a bootstrap approach to validate each model, so I performed a nested CV with an inner loop to tune the paramaters and an outer to have an unbiassed evaluation of the models.  

```{r, echo=FALSE}
Parameters = c('Interaction Depth','Boosting Iterations', 'Learning rate', 'Minimum leaf size'); Optimum_Value = c(6,100,0.010,10)
gbm.model = data.frame(Parameters, Optimum_Value)
kable(gbm.model, caption = 'Best Train Performing Boosting Model', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

Finally, I trained the optimum model with the training set and predicted the observations from the test set, leading to the following results:

```{r echo=FALSE}
Training_Acc = c(0.4883699); Test_Acc = c(0.4529412)
gbm.results = data.frame(Training_Acc, Test_Acc)
kable(gbm.results, caption = 'Boosting Performance', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

![Confusion Matrix Boosting](Images/cmboosting.png){align=center, heigth=100%, weigth=100%}

## 4. XGB  
\  
The **Extreme Gradient Boosting** uses the idea of the *boosting* algorithm, optimizing its learning with a *Cost function* to minimize. Thus this algorithm still uses the error of previous trees to develope new ones, yet has an addition restriction and is that the performance of a new trees has to be better than the performance of the previous in order to be used, trying to find the minimum value on a cost function. This algorithm has usually better accuracy with heterogeneous data. 

This model requires their variables to be numeric, thus I've compared two alternatives. First, I created some dummies out of the two qualitative variables. Secondly, as the qualitative variables are already factors I've transformed them in to *numeric* representing the level of the respective factor. Results showed that the second option was optimal with this specific data.

Using both data sets just described, I performed a *nestedCV* combined with a grid search to find the best model. The hyperparemeters are: **boosting iterations**, **max tree depth**, the **learning rate**, **minimum loss reduction**, subsample ratio of columns, minimum sum of instance weight and subsample percentage. Regarding the last 3 parameters I have mentioned I used the default values of 0.75, 0 and 1. 

```{r, echo=FALSE}
Parameters = c('Boosting Iterations', 'Max Tree Depth', 'Learning rate', 'Minimum loss reduction'); Optimum_Value = c(700,2,0.010,0.0001)
xgb.model = data.frame(Parameters, Optimum_Value)
kable(xgb.model, caption = 'Best Train Performing XGBoosting Model', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

Finally, I trained the optimum model with the trained set and predicted the observations from the test set, leading to the following results:

```{r echo=FALSE}
Data=c('Dummies', 'Númeric Factor');Training_Acc = c(0.4645881,0.4725473); Test_Acc = c(NA,0.5117647)
gbm.results = data.frame(Data, Training_Acc, Test_Acc)
kable(gbm.results, caption = 'XGB Performance', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

![Confusion Matrix XGB](Images/cmxgb.png){align=center, heigth=50%, weigth=50%}


## 5. KNN
\  
I decided to compare this algorithm too, as it uses a different approach compared to the tree-based algorithms. In this case the classification is done using proximity measures within the observations.  
The only tuning parameter here is the number of neighbors that two observation must have in order to consider them similar. Thus, I trained the algorithm with different *k* values and evaluated the performance with a *LOOCV* approach.  

![KNN Accuracy](Images/knn.png){align=center, heigth=100%, weigth=100%}

The *k* which showed better accuracy was  $k=20$, yet in the previous plot we see that $k=9$ already has a similar accuracy. With the latter model the performance was the following: 

```{r echo=FALSE}
Model = c(k=9); Training_Acc = c(0.3289086); Test_Acc = c(0.3823529)
gbm.results = data.frame(Model, Training_Acc, Test_Acc)
kable(gbm.results, caption = 'KNN Performance', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

![Confusion Matrix KNN](Images/cmknn.png){align=center, heigth=100%, weigth=100%}

## Best Model     

Comparing five different algorithms resulted in **Boosting** as the best one, i.e. with the highest training accuracy. With a simple Cross-Validation on all the training data resulted in following optimal parameters: 

```{r, echo=FALSE}
Parameters = c('Boosting Iterations', 'Max Tree Depth', 'Learning rate', 'Minimum loss reduction'); Optimum_Value = c(100,5,0.010,10)
xgb.model = data.frame(Parameters, Optimum_Value)
kable(xgb.model, caption = 'Best Train Performing XGBoosting Model', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

The final model classified the test set with an **accuracy of 55.30%**. 

# Conclusions  

```{r echo=FALSE}
kable(t(as.matrix(table(wildfires$Cause))), caption = 'Frecuency of the whole dataset', booktabs = T) %>% 
  kable_styling(latex_options = 'hold_position', font_size = 8)
```

![Confusion Matrix in proportion values, on the test data](Images/cm.prop.boosting.png){align=center, heigth=100%, weigth=100%}

The performance of the distinct models is pretty similar even though there is a large difference in complexity from *XGBoost* and *LDA* for instance. This similarity reflexes the complexity of the data, being 8 target categories in a relatively small set of 678 observations and 10 independent variables. Moreover, the information added by some variables is quite similar such as in the case of the *X* and *Y* coordinates and the *Square* which is a measure of location within a map. Also, the topic, predicting a wildfire's cause, is undeniably complex and requires many additional variables such as metrics describing the combustion of the vegetation of the area for instance. To sum up, all these factors lead to a similar performance among the algorithms.

Each model have a different confusion matrix, that is some algorithms classify better in some categories than others yet the *Boosting* model happened to have the highest proportion of true values. This model achieves to classify true positives in a wider range of categories, **XGB** and **LDA** had similar performance focused more in *Arson* and *Vegetation Burning*, whilst **Random Fores** and again **XGB** did a slightly better job on *Lighting*. Anyway, the model with highest proportion of true values on the training data is the **XGB**.

Analyzing the best performing model we have that the main predicted category is *Arson* followed by *Vegetation Burning* and *Unknown*. Whiles the categories with higher precision are *Arson* and *Vegetation Burning* and *Lightning*. My explanation for this fact is that, *Arson* clearly has the bast majority of the observations, actually, if a model predicts that all observations belong to this category the accuracy would already by roughly $39.38\%$. Then, regarding the other two, they are easily differentiated from the others as they usually occur in a specific time of the year.

Focusing on the **Boosting** model, from the confusion matrix we see that 4 of the classes have *0* true positives with an error rate of *1* that's probably due to the similarity between these categories; *Children*, *Engines*, *Miscellaneous*, *Smokers* and *Unknown.* Even though the miscellaneous category has more observations than the lightning one, the latter has a higher test accuracy that's probability because in the former category there is a mix of distinct unusual categories so it's extremely difficult to find a pattern, same with the *Unknown* one. The lack of accuracy predicting the fires originated by a child, an engine or a smoker can be associated with the model performance or the small amount of observation available for this three categories. 

Still with the best performing model, I've extracted the feature's importance with the proportional amount of gain given by each feature with the tree's splits.

#### Variable Importance      
\  
Relative influence that the boosting model associates to each variable.    

```{r echo=FALSE, fig.align='center', fig.pos='h', out.height="50%", out.width="75%", fig.cap='Proportion of Gain in splits by feature'}
knitr::include_graphics('Images/gbm_importance.png')
```

Probably a larger data set with a wider description of each observation would lead to a better classification of what did originate each fire. 

For the project's closures, I classified all observations corresponding to the **unknown observations** using a *Boosting* model, trained with the known observations.

```{r echo=FALSE, fig.align='center', fig.pos='h', out.height="30%", out.width="75%", fig.cap='Prediction of the Unkown observations'}
knitr::include_graphics('Images/pred.png')
```

