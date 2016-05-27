# README

## Python files

     main.py

### Description:
this is the top level file to start the training
input files:
	none
output files:
	none

     ild_helpers.py

### Description:
Helper file called in by main.py to load the data and other utilities functions

input files:
	   .pkl files
output files:
	   ILD_CNN_model.json
	   ILD_CNN_model_weights

     cnn_model.py

### Description:
model description, compilation and evaluation
input files:
	   .pkl files
output files:
	   ILD_CNN_model.json
	   ILD_CNN_model_weights
	   
	predict.py: 
### Description:
Fully stand alone file to predict label from dcom.
Generates bmp files from dicom for scan and lung mask(optional)
Generates patches from the scan images
Arrange patches for prediction machines
Actual prediction with CNN
Generate visualisation of predicted result in predict/<patient>/predicted_results
To run the full prediction, we need:
Inputs: the scan ang lung mask in "predict" directory as <patient> subdirectories, the CNN model and weights in "pickle" directory
Output: jpeg of scan images with predicted labels in "predict"/<patient>/"predicted_results"
predictlog.txt in "predict" with statistics on labels and error messages if any.

	genepatchFromScan.py
### Description:
Generate patches from dicom database.
Embeds all the sub routines, no need for other file
Input: dcm database in "HUG" / <top_level dir>(default ILD_TXT)
output: patch database, statistics on patches: label, localization, number,...
