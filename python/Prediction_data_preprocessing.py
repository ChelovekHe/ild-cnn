# coding: utf-8

# # Prepare predict dat for one full scan data
# ### Modified by Sylvain Kritter May 23, 2016
import g
import os
from scipy import misc
import numpy as np
import cPickle as pickle



for f in g.patient_list:
    print ('predict data processing work on: ',f)
    patient_dir_s = os.path.join(g.path_patient,f)

#    print(listcwd)
    patient_dir = os.path.join(patient_dir_s,g.patchpath)
    image_files = (os.listdir(patient_dir))
#    print image_files
    # creating variables
    # list for the merged pixel data
    dataset_list = []
    # list of the file reference data
    file_reference_list = []

    # go through all image files
    # 
    for fil in image_files:
#        print fil
        if fil.find(g.typei) > 0:  
#            print fil             
            # load the .bmp file into memory       
            image = misc.imread(os.path.join(str(patient_dir),fil), flatten= 0)        
            # append the array to the dataset list
            dataset_list.append(image)      
            # append the file name to the reference list. The objective here is to ensure that the data 
            # and the file information about the x/y position is guamarteed        
            file_reference_list.append(fil)
                
    # transform dataset list into numpy array                   
#    dataset = np.array(dataset_list)
#    X = dataset[:,:, :,1]
    X = np.array(dataset_list)
#    X = dataset[:,:, :,1]
    file_reference = np.array(file_reference_list)
    # this is already in greyscale 
#   

    #dir to put pickle files
    predictout_f_dir = os.path.join( patient_dir_s,g.picklefile)
    #print predictout_f_dir
    g.remove_folder(predictout_f_dir)
    os.mkdir(predictout_f_dir)

    xfp=os.path.join(predictout_f_dir,g.Xprepkl)
    xfpr=os.path.join(predictout_f_dir,g.Xrefpkl)
    pickle.dump(X, open( xfp, "wb" ))
    pickle.dump(file_reference, open( xfpr, "wb" ))
# 