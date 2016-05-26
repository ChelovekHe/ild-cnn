# README 

## Notebook files 


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


       ResultPlotting.ipynb
     
### Description:
Plotting loss functions and confusion matrices

issues: only fragment od functionality available so far


        ConfusionMatrix.ipynb 

### Description
Plotting confusion matrix tests


        visuapredict.ipynb  

### Description
notebook version of the visual prediction of the prediction results


      Prediction data preprocessing.ipynb

### Description:
takes a patients full set of patches from the predict directory and creates the prediction dataset and a side file with the file references
The current oatients ID is 121
    