# README

updated May 31, 2016

## Training 

     main.py

this is the top level file to start the training
input files:
	none
output files:
	none

## Helper Functions

     ild_helpers.py

Helper file called in by main.py to load the data and other utilities functions


## Model architecture and launch

     cnn_model.py

model description, compilation and evaluation


## Predict
	   
	predict.py: 

- Fully stand alone file to predict label from dcom.
- Generates bmp files from dicom for scan and lung mask(optional)
- Generates patches from the scan images
- Arrange patches for prediction machines
- Actual prediction with CNN
- Generate visualisation of predicted result in predict/<patient>/predicted_results
- To run the full prediction, we need:
	* Inputs: the scan ang lung mask in "predict" directory as <patient> subdirectories, the CNN model and weights in "pickle" directory
	* Output: bmp of scan images with predicted labels in "predict"/<patient>/"predicted_results" and ROI
	* predictlog.txt in "predict" with statistics on labels and error messages if any.
	* Parameter to enhance contrast (all in 0 255): "contrast" (default = True) at beginning of file
	* Add superposition of txt roi  from sroi directory if exists in visualization

## Patch generation

	genepatchFromScan.py

- Generate patches from dicom database.
- Embeds all the sub routines, no need for other file
- Input: dcm database in "HUG" / <top_level dir>(default ILD_TXT)
- output: patch database (both as per original :"patches' and normalized from 0 to 255:patches_norm) , statistics on patches: label, localization, number,...
- New directory "sroi" within each patient directory to store scan with ROI

