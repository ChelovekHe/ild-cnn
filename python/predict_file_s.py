# coding: utf-8
#Sylvain Kritter 15 Sept 2016
"""general parameters and file, directory names"""

import os
import cv2
import datetime
import dicom
import scipy
import shutil
import numpy as np
import ild_helpers as H
import cnn_model as CNN4
from keras.models import model_from_json

#########################################################
# for predict
# with or without bg (true if with back_ground)
wbg=False

#threshold for patch acceptance
thr = 0.9

#global directory for predict file
filedcm = '../predict_file/CT-2321-0011.dcm'
namedirtop='predict_file'

#subdirectory name to put images
jpegpath = 'patch_jpeg'

#directory with lung mask dicom
lungmask='lung_mask'

#directory to put  lung mask bmp
lungmaskbmp='bmp'

#directory name with scan with roi if exists
sroi='sroi'

#directory with bmp from dicom
scanbmp='scan_bmp'

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
#print path_patient
#########################################################

#color of labels

red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)

# label we consider, number will start at 0 anyway
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


classifc ={
'back_ground':darkgreen,
'consolidation':red,
'fibrosis':blue,
'ground_glass':yellow,
'healthy':green,
'micronodules':cyan,
'reticulation':purple,

'air_trapping':white,
 'bronchial_wall_thickening':white,
 'bronchiectasis':white,
 'cysts':white,
 'early_fibrosis':white,
 'emphysema':white,
 'increased_attenuation':white,
 'macronodules':white,
 'pcp':white,
 'peripheral_micronodules':white,
 'tuberculosis':white
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
#     print(tabi.min(), tabi.max())
     tabi1=tabi-tabi.min()
#     print(tabi1.min(), tabi1.max())
     tabi2=tabi1*(255/float(tabi1.max()-tabi1.min()))
#     print(tabi2.min(), tabi2.max())
     return tabi2 
    


def pavgene (data):
    """ generate patches from scan"""
    print('generate patches on: ',data)
#    print data
    patch_list=[]

    endnumslice=data.find('.dcm')
    posend=endnumslice
    while data.find('-',posend)==-1:
        posend-=1
    debnumslice=posend+1
#    print debnumslice ,endnumslice
    slicenumberscan=int(data[debnumslice:endnumslice])
#    print slicenumberscan    
    (top,tail)=os.path.split(data)
   
#    print 'tail' ,tail
    endnumslice=tail.find('.dcm')    
    core=tail[0:endnumslice]
    corefull=core+'.bmp'
    imagedir=os.path.join(bmp_dir,corefull)
#    (top,tail)=os.path.split(top)
      
    
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
                                        
#                cv2.imshow('image',img2)
#                cv2.waitKey(0)
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
#              
        # convention img[y: y + h, x: x + w]
#                  
#                    cv2.imshow('image',crop_img)
#                    cv2.waitKey(0)
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
#                                normpatch = cv2.equalizeHist(imgray)
                                na=np.array(imgray)
                                normpatch=normi(na)  
                                patch_list.append((slicenumberl,x,y,normpatch))
                            
                        #                print('pavage',i,j)  
                                i=0
                                while i < dimpavx:
                                    j=0
                                    while j < dimpavy:
                                        if y+j<512 and x+i<512:
                                            tabp[y+j][x+i]=red
                                        if i == 0 or i == dimpavx-1 :
                                            j+=1
                                        else:
                                            j+=dimpavy-1
                                    i+=1
                            #we cancel the source                            
                                thresh[y:y+dimpavy, x:x+dimpavx]=0
                                
                                y+=dimpavy-1                                                  
                        y+=1
                    x+=1
            
            tabp =imglung+tabp

            break
    return patch_list

# 
def ILDCNNpredict(patch_list)   :  
        dataset_list=[]
        for fil in patch_list:

            dataset_list.append(fil[3])

        X = np.array(dataset_list)
        X_predict = np.asarray(np.expand_dims(X,1))/float(255)        
        
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
#        model = H.load_model()

        model = model_from_json(open(jsonf).read())
        model.load_weights(weigf)

        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        

        proba = model.predict_proba(X_predict, batch_size=100)

        return proba


def dd(i):
    if (i)<10:
        o='0'+str(i)
    else:
        o=str(i)
    return o
    
t = datetime.datetime.now()

today = str('date: '+dd(t.month)+'-'+dd(t.day)+'-'+str(t.year)+\
'_'+dd(t.hour)+':'+dd(t.minute)+':'+dd(t.second))
print today

def doPrediction(data):
#    print data
    genebmp(data)
    plist=pavgene(data)
#    dataprocessing(data)
    proba=ILDCNNpredict(plist)
    return proba,plist

#####################################################################
#all directories
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#print cwdtop
path_patient = os.path.join(cwdtop,namedirtop)

bmp_dir = os.path.join(path_patient, scanbmp)
remove_folder(bmp_dir)    
os.mkdir(bmp_dir)  

lung_dir = os.path.join(path_patient, lungmask)
if os.path.exists(lung_dir)== False:
    os.mkdir(lung_dir)
    
lung_dir_bmp=os.path.join(lung_dir, lungmaskbmp)
remove_folder(lung_dir_bmp)    
os.mkdir(lung_dir_bmp)

picklein_file = os.path.join(path_patient,picklefile_source)


dirpatientfsdb=os.path.join(path_patient,sroi)

##########################################################
print('work on:',filedcm)
#print filedcm

(p,l)=doPrediction(filedcm)
#    print namedirtopcf
        
print('completed on: ',filedcm)
        
t = datetime.datetime.now()
today = str('date: '+dd(t.month)+'-'+dd(t.day)+'-'+str(t.year)+\
'_'+dd(t.hour)+':'+dd(t.minute)+':'+dd(t.second))
print today

