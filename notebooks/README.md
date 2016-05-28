# README 

## Notebook files
updated: May 28, 2016 


     TrainingDataCreation.ipynb

### Description:
notebook that takes the patches and creates the dataset.
Amongst the total amount of patches, uses patches only in chunks of 1000. 
Selection of patches is random based. 
Split the dataset into training, validation and test set
Saves the sets into the pickl directory in 6 files


      VisualCheckingPatches.ipynb
     
### Description:
review patches in batches of 20 or individually


       ShowPatientResults.ipynb
     
### Description:
The ID of a patient is chosen and then you can browse through the jpeg images with the results.


       ResultPlotting.ipynb
     
### Description:
Plotting loss functions and confusion matrices
Not complete

issues: only fragment od functionality available so far


        ConfusionMatrix.ipynb 

### Description
Plotting confusion matrix tests


        visuapredict.ipynb  

### Description

don't use! not complete
notebook version of the visual prediction of the prediction results


      Prediction data preprocessing.ipynb

### Description:
takes a patients full set of patches from the predict directory and creates the prediction dataset and a side file with the file references
The current oatients ID is 121
    