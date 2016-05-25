# coding: utf-8
#Sylvain Kritter 24 mai 2016
"""general parameters and file, directory names"""
import os
import shutil

cwd=os.getcwd()
#######################################################
#customisation part
# define the dicom image format bmp or jpg

#global directory for predict file
namedirtop = 'predict'

#directory for storing image out after prediction
predictout='predicted_results'
#directory for patches from scan images
patchpath='patch_bmp'

#subdirectory name to put images
jpegpath = 'patch_jpeg'

#directory with lung mask dicom
lungmask='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='bmp'

#directory with bmp from dicom
scanbmp='scan_bmp'

#image  patch format
typei='bmp' #can be jpg
#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx =32
dimpavy = 32

#threshold for probability prediction
thrproba = 0.8

#threshold for patch acceptance
thrpatch = 0.8

#pickle with predicted classes
predicted_classes = 'predicted_classes.pkl'

#pickle with predicted probabilities
predicted_proba= 'predicted_probabilities.pkl'
#pickle with Xfile
Xprepkl='X_predict.pkl'
Xrefpkl='X_file_reference.pkl'


#subdirectory name to colect pkl files resulting from prediction
picklefile='pickle'

#end customisation part
#########################################################
(cwdtop,tail)=os.path.split(cwd)
path_patient = os.path.join(cwdtop,namedirtop)
#print PathDicom
patient_list= os.walk(path_patient).next()[1]
#print patient_list

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

def interv(borne_inf, borne_sup):
    """Générateur parcourant la série des entiers entre borne_inf et borne_sup.
    inclus
    Note: borne_inf doit être inférieure à borne_sup"""
      
    while borne_inf <= borne_sup:
        yield borne_inf
        borne_inf += 1