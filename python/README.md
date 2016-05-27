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
Fully stand alone file to predict label from dcom.
	-generates bmp files from dicom for scan and lung mask(optional)
	-generates patches from the scan images
	-arrange patches for prediction machines
	-actual prediction with CNN
	-generate visualisation of predicted result in predict/<patient>/predicted_results
	C
To run the full prediction, we need:
Inputs
	-the scan ang lung mask in "predict" directory as <patien> subdirectories
	-the CNN model and weights in "pickle" directory
Output:
jpeg of scan images with predicted labels in preedict/<patient>/predicted_results
predictlog.txt in <predict> with statistics on labels and error messages if any
