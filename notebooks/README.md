# README 

### Notebook files
updated: May 31, 2016 

## Dataset Creation for training

     TrainingDataCreation.ipynb

Takes the .bmp files from the patches directory and creates the dataset.

Augments the data by rotating the patches 3 times by 90 degrees

Amongst the total amount of patches, uses patches only in chunks of 1000. 

Selection of patches is random based. 

Split the dataset into training (50%), validation (25%) and test set (25%).

Saves the sets into the pickle directory in 6 files
X_train, X_val, X_test for the data and y_train, y_val, y_test for labels


## Visual Check of patches

      VisualCheckingPatches.ipynb
     
review patches in batches of 20 or individually


## Prediction results per patient

       ShowPatientResults.ipynb
     
The ID of a patient is chosen and then you can browse through the jpeg images with the results.


## Training results #1 

       ResultPlotting.ipynb
     
Plots various variables over training session.

Currently 
- f-score on validation sets
- accuracy on validation sets
- loss functions on Training set and validation set



## Training results 2

        ConfusionMatrix.ipynb 

Plotting confusion matrix of the test set


 
    