# coding: utf-8
#Sylvain Kritter 11 septembre 2016
"""general parameters and file, directory names"""

import os
import cv2

import dicom
import scipy
from scipy import misc
import shutil
import numpy as np

import cPickle as pickle
import ild_helpers as H
import cnn_model as CNN4
from keras.models import model_from_json


#########################################################
# for predict
# with or without bg (true if with back_ground)
wbg=False
#to enhance contrast on patch put True
contrast=True
#threshold for patch acceptance
thr = 0.9

#global directory for predict file
filedcm = '../predict_file/CT-2321-0011.dcm'
namedirtop='predict_file'

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


#pickle with predicted probabilities
predicted_proba= 'predicted_probabilities.pkl'
#pickle with Xfile
Xprepkl='X_predict.pkl'
Xrefpkl='X_file_reference.pkl'

#subdirectory name to colect pkl files resulting from prediction
picklefile_dest='pickle_dest'
#subdirectory name to colect weights
picklefile_source='pickle_source'


#end predict part
#########################################################
# general

#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx =32
dimpavy = 32

pxy=float(dimpavx*dimpavy)*255

#end general part

#########################################################



#all the possible labels
classifstart ={
'back_ground':0,
'consolidation':1,
'fibrosis':2,
'ground_glass':3,
'healthy':4,
'micronodules':5,
'reticulation':6,

'air_trapping':7,
 'bronchial_wall_thickening':8,
 'bronchiectasis':9,
 'cysts':10,
 'early_fibrosis':11,
 'emphysema':12,
 'increased_attenuation':13,
 'macronodules':14,
 'pcp':15,
 'peripheral_micronodules':16,
 'tuberculosis':17
  }

#only label we consider, number will start at 0 anyway
if wbg :
    classif ={
    'back_ground':0,
    'consolidation':1,
    'fibrosis':2,
    'ground_glass':3,
    'healthy':4,
    'micronodules':5,
    'reticulation':6,    
    'air_trapping':7,
     'bronchial_wall_thickening':8,
     'bronchiectasis':9,
     'cysts':10,
     'early_fibrosis':11,
     'emphysema':12,
     'increased_attenuation':13,
     'macronodules':14,
     'pcp':15,
     'peripheral_micronodules':16,
     'tuberculosis':17
      }
else:
     classif ={
    'consolidation':0,
    'fibrosis':1,
    'ground_glass':2,
    'healthy':3,
    'micronodules':4,
    'reticulation':5, 
    'air_trapping':6,
     'bronchial_wall_thickening':7,
     'bronchiectasis':8,
     'cysts':9,
     'early_fibrosis':10,
     'emphysema':11,
     'increased_attenuation':12,
     'macronodules':13,
     'pcp':14,
     'peripheral_micronodules':15,
     'tuberculosis':16
      }





def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

   
def genebmp(data):
    """generate patches from dicom files"""
    print ('load dicom files in :',data)

    RefDs = dicom.read_file(data) 
    (top,tail)=os.path.split(data)
   
    endnumslice=tail.find('.dcm')
     
    core=tail[0:endnumslice]
#    print core
    posend=endnumslice
    while tail.find('-',posend)==-1:
        posend-=1
    debnumslice=posend+1
    slicenumber=int(tail[debnumslice:endnumslice])

    filebmp=core+'.bmp'
    filetw=os.path.join(bmp_dir,filebmp)

    scipy.misc.imsave(filetw, RefDs.pixel_array)
    
#generate bmp for lung    
    
     
    listlung=os.listdir(lung_dir)
#    print listlung
    lungexist=False
    for l in listlung:
          if ".dcm" in l.lower():
             endnumslice=l.find('.dcm')    
             corelung=l[0:endnumslice]
             lungexist=True
             break
#    print corelung
    if lungexist:
        posend=endnumslice
        while l.find('_',posend)==-1:
            posend-=1
        debnumslice=posend+1
        slicenumberlung=int(l[debnumslice:endnumslice])
    #    print slicenumberlung
        if slicenumber== slicenumberlung:
            lungfile=os.path.join(lung_dir,l)
            RefDslung = dicom.read_file(lungfile)
            filebmplung=corelung+'.bmp'
            filetwlung=os.path.join(lung_dir_bmp,filebmplung)
            scipy.misc.imsave(filetwlung, RefDslung.pixel_array)
    else:
            tablung = np.ones((dimtabx, dimtaby), dtype='i')
            tablung[0:dimtaby-1, 0:dimtabx-1]=255
            filebmplung=core+'_'+str(slicenumber)+'.bmp'
            filetwlung=os.path.join(lung_dir_bmp,filebmplung)
            scipy.misc.imsave(filetwlung, tablung)
            
   
   
def normi(img):
     """ normalise patches 0 255"""
     tabi = np.array(img)

     tabi1=tabi-tabi.min()

     tabi2=tabi1*(255/float(tabi1.max()-tabi1.min()))

     return tabi2 
    


def pavgene (data):
    """ generate patches from scan"""
    print('generate patches on: ',data)
#    print data

    endnumslice=data.find('.dcm')
    posend=endnumslice
    while data.find('-',posend)==-1:
        posend-=1
    debnumslice=posend+1
#    print debnumslice ,endnumslice
    slicenumberscan=int(data[debnumslice:endnumslice])
#    print slicenumberscan    
    (top,tail)=os.path.split(data)
   
#    print tail
    endnumslice=tail.find('.dcm')    
    core=tail[0:endnumslice]
    corefull=core+'.bmp'
    imagedir=os.path.join(bmp_dir,corefull)
         
    listlung=os.listdir(lung_dir_bmp)
    for  n in listlung:    
            tabp = np.zeros((dimtabx, dimtaby,3), dtype='i')
          
            endnumslice=n.find('.bmp')           
            posend=endnumslice
#            print n
            while n.find('_',posend)==-1:
                posend-=1
            debnumslice=posend+1
            slicenumberl=int((n[debnumslice:endnumslice])) 
#            print slicenumberl
            if slicenumberscan==slicenumberl:

                lungfile = os.path.join(lung_dir_bmp, n)
#                print lungfile
                imglung=cv2.imread(lungfile,1)
                img1 = cv2.imread(imagedir,1)
                img2 = cv2.medianBlur(imglung,9)
#                cv2.imshow('image',img1)
#                cv2.waitKey(0)
                imgray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                ret,thresh = cv2.threshold(imgray,0,255,0)
                atabf = np.nonzero(thresh)
                imagemax= cv2.countNonZero(imgray)

                if  imagemax>0:
#            print thresh.size
                    xmin=atabf[1].min()
                    xmax=atabf[1].max()
                    ymin=atabf[0].min()
                    ymax=atabf[0].max()
                else:
                    xmin=0
                    xmax=20
                    ymin=0
                    ymax=20
            
                x=xmin
                nbp=0
                while x <= xmax:
                    y=ymin
                    while y<=ymax:
                        crop_img = thresh[y:y+dimpavy, x:x+dimpavx]              
        # convention img[y: y + h, x: x + w]

                        area= crop_img.sum()
#         
                        targ=float(area)/pxy
#                    print targ, area ,pxy
                        if targ >thr:
#                        print targ, area ,pxy
                            crop_img_orig = img1[y:y+dimpavy, x:x+dimpavx]   
                        
                            imgray = cv2.cvtColor(crop_img_orig,cv2.COLOR_BGR2GRAY)
                            imagemax= cv2.countNonZero(imgray)
                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)
#                        print imagemax

                     
                            if imagemax>0 and min_val!=max_val:

                                nbp+=1
                                nampa='p_'+str(slicenumberscan)+'_'+str(x)+'_'+str(y)+'.bmp'                            
#                                normpatch = cv2.equalizeHist(imgray)
                                na=np.array(imgray)
                                normpatch=normi(na)
                                fw=os.path.join(patchpathdir,nampa)                               
                                cv2.imwrite(fw,normpatch)                           
                           
                            #we cancel the source                         
                                thresh[y:y+dimpavy, x:x+dimpavx]=0                            
                                y+=dimpavy-1
            
                        y+=1
                    x+=1
            
            tabp =imglung+tabp

            scipy.misc.imsave(jpegpathdir+'/s_'+str(slicenumberscan)+'.jpg', tabp)
            break



def dataprocessing(data):
    
    print ('generate data for CNN on: ',data)

    # list for the merged pixel data
    dataset_list = []
    # list of the file reference data
    file_reference_list = []
    image_files = (os.listdir(patchpathdir))
    # go through all image files
    # 
    for fil in image_files:
#        print fil
        if fil.find('.bmp') > 0:  
#            print fil             
            # load the .bmp file into memory       
            image = misc.imread(os.path.join(str(patchpathdir),fil), flatten= 0)        
            # append the array to the dataset list
            dataset_list.append(image)      
            # append the file name to the reference list. The objective here is to ensure that the data 
            # and the file information about the x/y position is guamarteed        
            file_reference_list.append(fil)
                
    # transform dataset list into numpy array                   

    X = np.array(dataset_list)
    # this is already in greyscale 
    file_reference = np.array(file_reference_list)
#   
    #dir to put pickle files

    xfp=os.path.join(predictout_f_dir,Xprepkl)
    xfpr=os.path.join(predictout_f_dir,Xrefpkl)
    pickle.dump(X, open( xfp, "wb" ))
    pickle.dump(file_reference, open( xfpr, "wb" ))
# 
def ILDCNNpredict(data):     
        print ('predict patches on: ',data) 
     
        jsonf= os.path.join(picklein_file,'ILD_CNN_model.json')
#        print jsonf
        weigf= os.path.join(picklein_file,'ILD_CNN_model_weights')
#        print weigf
#model and weights fr CNN
        args  = H.parse_args()                          
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

        model = model_from_json(open(jsonf).read())
        model.load_weights(weigf)
        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        
#
        patient_pred_file =os.path.join( predictout_f_dir,Xprepkl)
        X_predict = pickle.load( open( patient_pred_file, "rb" ) )

    # adding a singleton dimension and rescale to [0,1]
        X_predict = np.asarray(np.expand_dims(X_predict,1))/float(255)

    # predict and store  classification and probabilities 
        proba = model.predict_proba(X_predict, batch_size=100)
    # store  classification and probabilities 
        xfproba=os.path.join( predictout_f_dir,predicted_proba)
        pickle.dump(proba, open( xfproba, "wb" ))


def doPrediction(data):
#    print data
    genebmp(data)
    pavgene(data)
    dataprocessing(data)
    ILDCNNpredict(data)


#all directories
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#print cwdtop
path_patient = os.path.join(cwdtop,namedirtop)

bmp_dir = os.path.join(path_patient, scanbmp)
remove_folder(bmp_dir)    
os.mkdir(bmp_dir)  

patchpathdir = os.path.join(path_patient,patchpath)
remove_folder(patchpathdir)    
os.mkdir(patchpathdir) 

jpegpathdir = os.path.join(path_patient,jpegpath)
remove_folder(jpegpathdir)    
os.mkdir(jpegpathdir)

lung_dir = os.path.join(path_patient, lungmask)
if os.path.exists(lung_dir)== False:
    os.mkdir(lung_dir)
lung_dir_bmp=os.path.join(lung_dir, lungmaskbmp)
remove_folder(lung_dir_bmp)    
os.mkdir(lung_dir_bmp)
#    print 'patchpathdir:,',patchpathdir
   
picklein_file = os.path.join(path_patient,picklefile_source)
predictout_f_dir = os.path.join(path_patient,picklefile_dest)
remove_folder(predictout_f_dir)    
os.mkdir(predictout_f_dir)


print('work on:',filedcm)
#print filedcm

doPrediction(filedcm)


