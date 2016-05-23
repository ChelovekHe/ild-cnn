# ILD_CNN_prediction_full_flow

# May 22, 2016

# using patient 121 who has fibrosis
# as reference, patient 138 is healthy
# patient 107 has both reticulation and groundglass

patient_ID = 121



import os
from scipy import misc
import numpy as np

from six.moves import cPickle
import sys
import cPickle as pickle
import cv2
import argparse
import json
from keras.models import model_from_json

from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.utils import np_utils 
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU


# sys.path.insert(0, '../python')
# taken out as current directory is alread python

import ild_helpers as H
import cnn_model as CNN

one_folder_up = os.path.dirname(os.getcwd())
print one_folder_up

# create path to patch directory
predict_dir = os.path.join(one_folder_up, 'predict')

# list all directories under patch directory. They are representing the categories
patient_list = (os.listdir(predict_dir))
print 'taking out the item : ', patient_list.pop(0)

# create path to patient directory
patient_dir = os.path.join(predict_dir, str(patient_ID))
patient_dir

# list all directories under patch directory. They are representing the categories
image_files = (os.listdir(patient_dir))
len(image_files)

print 'sampling from this list'
image_files[0:5]

# creating variables
# list for the merged pixel data
dataset_list = []
# list of the file reference data
file_reference_list = []


# go through all image files
# 


for file in image_files:
                
    if file.find('.bmp') > 0:
                
        # load the .bmp file into memory       
        image = misc.imread(os.path.join(str(patient_dir),file), flatten= 0)
        
        # append the array to the dataset list
        dataset_list.append(image)
        
        # append the file name to the reference list. The objective here is to ensure that the data 
        # and the file information about the x/y position is guamarteed
        
        file_reference_list.append(file)
                
                
                                 
# transform dataset list into numpy array                   
dataset = np.array(dataset_list)
file_reference = np.array(file_reference_list)

# use only one of the 3 color channels as greyscale info
X_predict = dataset[:,:, :,1]

print 'dataset X shape is now: ', X_predict.shape
print 'X_file_reference list for the first 5 items is : ' 
print file_reference[0:1]
print file_reference[1:2]
print file_reference[2:3]
print file_reference[3:4]
print file_reference[4:5]

print X_predict.shape
print file_reference.shape

args         = H.parse_args()                          
train_params = {
     'do' : float(args.do) if args.do else 0.5,        
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 5,         # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res'      # csv results filename alias
}

# load both the model and the weights 

model = H.load_model()

model.compile(optimizer='Adam', loss=CNN.get_Obj(train_params['obj']))

# adding a singleton dimension and rescale to [0,1]

X_predict = np.asarray(np.expand_dims(X_predict,1))/float(255)

# predict and check classification and probabilities are the same

classes = model.predict_classes(X_predict, batch_size=10)
proba = model.predict_proba(X_predict, batch_size=10)

print classes
print proba

# generate the pickl file name with the patient ID suffix

file_name_classes =  '../pickle/' + 'predicted_classes' + '_' + str(patient_ID) + '.pkl'
file_name_probabilities =  '../pickle/' + 'predicted_probabilities' + '_' + str(patient_ID) + '.pkl'
print file_name_classes
print file_name_probabilities

print 'prediction completed'

