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
	   
	g.py :
contents all the directory and file names

	genebmpforpredict.py: 
generates bmp files from dicom for scan and lung mask

	prepascanmask.py: 
generates patches from the scan images

Prediction_data_preprocessing.py: 
arrange patches for prediction machines

	ILD_CNN_Predict.py: 
actual prediction

	visuapredict.py: 
	generate visualisation of predicted result in predict/<patient>/predicted_results
Creates also a visu_errorfile.txt in <predict> with statistics on labels.

	predictionpython.bat: 
batch file for window

To rune the full prediction, we need:
-the scan ang lung mask in predict directory
-run in the order:
python genebmpforpredict.py
python prepascanmask.py
python Prediction_data_preprocessing.py
python ILD_CNN_Predict.py
python visuapredict.py
