# Wildfires Classification

Project based on the application of distinct classification algorithms in order to determine the cause of wildfires.


## Models

+ LDA, with a previous PCA to train with the principal components
+ Tree Based Models: 
  + Random Forest
  + Boosting
  + XGBoosting
+ KNN

## Results 

**Boosting** has been the best performing model. The following images show the confusion matrix of the test set prediction, and the importance of the model's features for the best model.

<p align = "center">
<img src="Images/cm.prop.boosting.png" width="250" height="200">
</p>

<p align = "center">
<img src="Images/imp_gain.png" width="250" height="200">
 </p>
