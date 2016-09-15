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

- Fully stand alone file to predict label from dicom.
- Generates bmp files from dicom for scan and lung mask(optional)
- Generates patches from the scan images
- Arrange patches for prediction machines
- Actual prediction with CNN
- Generate visualization of predicted result in predict/<patient>/predicted_results
- To run the full prediction, we need:
	* Inputs: the scan lung mask in "predict" directory as <patient> subdirectories, the CNN model and weights in "pickle" directory
	* Output: bmp of scan images with predicted labels in "predict"/<patient>/"predicted_results" and ROI
	* predictlog.txt in "predict" with statistics on labels and error messages if any.
	* scan image with superposition of txt roi  from sroi directory if exists in visualization

## Patch generation

	genepatchFromScan.py

- Generate patches from dicom database.
- Embeds all the sub routines, no need for other file
- Input: dcm database in "HUG" / <top_level dir>(default ILD_TXT)
- output: patch database (both as per original :"patches' and normalized from 0 to 255:patches_norm) , statistics on patches: label, localization, number,...
- New directory "sroi" within each patient directory to store scan with ROI

## Prediction for one dicom at a time

	predict_file.py

- generate predicted probabilities for one dicom file
- main inputs: (all as variables at top of py file)
	* name of dicom file in variable: "filedcm" (complete path)
	* directory where this file is in "namedirtop" variable. note: several directories will be created in it
	* lung mask data: in "lung_mask" directory for dicom
	* model and weights in "pickle_source" directories
	* a switch to define if back-ground is used or not : "wbg" (False by default)
	* size of dicom (512x512) and patches (32x32)
	* list of classes with their number (must start at 0) in dictionary "classif"
- main outputs:
	* scan image in bmp format with all patches as overlay, with average probability by class in directory "predicted_result"
	* scan in jpg format with all patches as overlay, in "patch_jpeg"
	* scan image converted in bmp format in directory "scan_bmp"
	* lung mask in bmp format in dirctory "lung_mask/bmp"

## Prediction for one dicom at a time, without visualization

	predict_file_s.py

-same that predict_file.py, without visualization, only usefull lines
- generate predicted probabilities for one dicom file
- main inputs: (all as variables at top of py file)
	* name of dicom file in variable: "filedcm" (complete path)
	* directory where this file is in "namedirtop" variable. Note: several directories will be created in it
	* lung mask data: in "lung_mask" directory for dicom
	* model and weights in "pickle_source" directories
	* a switch to define if back-ground is used or not : "wbg" (False by default)
	* size of dicom (512x512) and patches (32x32)
	* list of classes with their number (must start at 0) in dictionary "classif", compatible with model of CNN
- main outputs:
	* doPrediction function returns 2 variables: "p" and "l"
	* variable "p":this is an np array of length n, n being the number of patches, of c-tuples with probabilities ordered by classes number, c being the number of classes, defined in the model in "pickle_source" directory)
	* variable "l: this is an np array of length n with the list of patches in the format: (slice number, upper left corner x, upper left corner y, patch image as array)
	* scan image converted in bmp format in directory "scan_bmp"
	* lung mask in bmp format in directory "lung_mask/bmp"

