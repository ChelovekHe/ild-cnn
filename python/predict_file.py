# coding: utf-8
#Sylvain Kritter 24 mai 2016
"""general parameters and file, directory names"""

import os
import cv2
import datetime
import time
import dicom
import scipy
from scipy import misc
import shutil
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw
import cPickle as pickle
import ild_helpers as H
import cnn_model as CNN4
from keras.models import model_from_json
import numpy

#########################################################
# for predict
# with or without bg (true if with back_ground)
wbg=False
#to enhance contrast on patch put True
contrast=True
#threshold for patch acceptance
thr = 0.9
#threshold for probability prediction

#global directory for predict file
filedcm = '../predict_file/CT-2321-0011.dcm'
namedirtop='predict_file'

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
#directory name with scan with roi
sroi='sroi'
#directory with bmp from dicom
scanbmp='scan_bmp'
#directory for bmp from dicom
#bmpname='bmp'

#pickle with predicted classes
#predicted_classes = 'predicted_classes.pkl'

#pickle with predicted probabilities
predicted_proba= 'predicted_probabilities.pkl'
#pickle with Xfile
Xprepkl='X_predict.pkl'
Xrefpkl='X_file_reference.pkl'

#subdirectory name to colect pkl files resulting from prediction
picklefile_dest='pickle_dest'
#subdirectory name to colect weights
picklefile_source='pickle_source'



#print PathDicom
#patient_list= os.listdir(path_patient)
#print patient_list
# list label not to visualize
#excluvisu=['back_ground','healthy']
excluvisu=[]

#dataset supposed to be healthy
#datahealthy=['138']
#end predict part
#########################################################
# general
#image  patch format
#typei='bmp' #can be jpg
#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx =32
dimpavy = 32

pxy=float(dimpavx*dimpavy)*255

#end general part
#font file imported in top directory
font20 = ImageFont.truetype( 'arial.ttf', 20)
font10 = ImageFont.truetype( 'arial.ttf', 10)
#print path_patient
#########################################################
#errorfile = open(cwdtop+'/predictlog.txt', 'w') 

#color of labels


red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)



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

#align label to 0 for compatibility
#minc=1000
#for f in classif:
#    if classif[f] < minc:
#        minc=classif[f]
#        
#for f in classif:
#   classif[f] =classif[f]-minc
##print classif


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

                     
                            if imagemax==0 or min_val==max_val:

                                errortext='uniform pixel in: '+ data                           
#                            print(errortext)
#                            print (min_val, max_val,min_loc, max_loc)
#                          
                            else:
                                nbp+=1
                                nampa='p_'+str(slicenumberscan)+'_'+str(x)+'_'+str(y)+'.bmp'                            
#                                normpatch = cv2.equalizeHist(imgray)
                                na=np.array(imgray)
                                normpatch=normi(na)
                                fw=os.path.join(patchpathdir,nampa)                               
                                cv2.imwrite(fw,normpatch)                           
                            
                        #                print('pavage',i,j)  
                                i=0
                            #we draw the rectange
                                
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
                           
#                            cv2.imshow('image',imglung+tabp)
#                            cv2.waitKey(0)
                                                  
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
#    dataset = np.array(dataset_list)
#    X = dataset[:,:, :,1]
    X = np.array(dataset_list)
    # this is already in greyscale 
#    X = dataset[:,:, :,1]
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
#        model = H.load_model()

        model = model_from_json(open(jsonf).read())
        model.load_weights(weigf)

        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        
#

        patient_pred_file =os.path.join( predictout_f_dir,Xprepkl)
#        print patient_pred_file
        X_predict = pickle.load( open( patient_pred_file, "rb" ) )
#    print X_predict
    # adding a singleton dimension and rescale to [0,1]
        X_predict = np.asarray(np.expand_dims(X_predict,1))/float(255)

    # predict and store  classification and probabilities 
#        classes = model.predict_classes(X_predict, batch_size=10)
        proba = model.predict_proba(X_predict, batch_size=10)
    # store  classification and probabilities 
#        xfc=os.path.join( patient_dir_pkl,predicted_classes)
        xfproba=os.path.join( predictout_f_dir,predicted_proba)
#        pickle.dump(classes, open( xfc, "wb" ))
        pickle.dump(proba, open( xfproba, "wb" ))

def fidclass(numero):
    """return class from number"""
    found=False
    for cle, valeur in classif.items():
        
        if valeur == numero:
            found=True
            return cle
      
    if not found:
        return 'unknown'


def tagview(fig,label,pro,x,y):
    """write text in image according to label and color"""
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=classifc[label]
    labnow=classifstart[label]-1
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltax=0
        deltay=60
    else:        
        deltay=25*((labnow)%3)
        deltax=175*((labnow)//3)
#    print (x+deltax,y+deltay)
    #print text, col
    draw.text((x+deltax, y+deltay),label+' '+pro,col,font=font20)

    imgn.save(fig) 
    
def tagviews(fig,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.rectangle ([x1, y1,x1+100, y1+15],outline='black',fill='black')
    draw.text((x0, y0),t0,white,font=font10)
    draw.text((x1, y1),t1,white,font=font10)
    draw.text((x2, y2),t2,white,font=font10)
    draw.text((x3, y3),t3,white,font=font10)
    draw.text((x4, y4),t4,white,font=font10)
    imgn.save(fig)

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m


def loadpkl(dop):
    """crate image directory and load pkl files"""

    #pickle with predicted classes
#    preclasspick= os.path.join(dop,predicted_classes)
    #pickle with predicted probabilities
    preprobpick= os.path.join(dop,predicted_proba)
     #pickle with xfileref
    prexfilepick= os.path.join(dop,Xrefpkl)
    """generate input tables from pickles"""

    dd = open(preprobpick,'rb')
    my_depickler = pickle.Unpickler(dd)
    preprob = my_depickler.load()
    dd.close()  
    dd = open(prexfilepick,'rb')
    my_depickler = pickle.Unpickler(dd)
    prexfile = my_depickler.load()
    dd.close()  
#    return (preclass,preprob,prexfile)
    return (preprob,prexfile)


def addpatch(col,lab, xt,yt):
    imgi = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    colr=[col[2],col[1],col[0]]
#    numl=listlabel[lab]
    tablint=[(xt,yt),(xt,yt+dimpavy),(xt+dimpavx,yt+dimpavy),(xt+dimpavx,yt)]
    tabtxt=np.asarray(tablint)
#    print tabtxt
    cv2.polylines(imgi,[tabtxt],True,col)
    cv2.fillPoly(imgi,[tabtxt],col)
    return imgi

def drawContour(imi,ll):
    
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
    for l in ll:
#        print l
        col=classifc[l]

        masky=cv2.inRange(imi,col,col)
        outy=cv2.bitwise_and(imi,imi,mask=masky)
        imgray = cv2.cvtColor(outy,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,0,255,0)
        im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)        
        contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
#        cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
        cv2.drawContours(vis,contours,-1,col,1)

    return vis
#cv2.drawContours(im,contours,-1,(0,255,0),-1)
        

def  visua(data):
    print('image generation from predict: ',data)
    (top,tail)=os.path.split(data)
#    print tail
    endnumslice=tail.find('.dcm')    
    core=tail[0:endnumslice]
    #directory name with predict out dabasase, will be created in current directory
    
    (preprob,listnamepatch)=loadpkl(predictout_f_dir)
    
    listbmpscan=os.listdir(bmp_dir)
    listlabelf={}
    print listbmpscan
#    setname=f
#    tabsim1 = np.zeros((dimtabx, dimtaby), dtype='i')
    for img in listbmpscan:
        print img
        #assume less than 100 patches per slice
        imgt = np.zeros((dimtabx,dimtaby,3), np.uint8)
        imn = np.zeros((dimtabx,dimtaby,3), np.uint8)
        listlabel={}
        listlabelaverage={}
        if os.path.exists(dirpatientfsdb):
#        imgc=os.path.join(dirpatientfdb,img)
            imgc=os.path.join(dirpatientfsdb,img)
        else:
            imgc=os.path.join(bmp_dir,img)

        
#                        orig= Image.open(lungcoref,'r')
#                        ncr=orig.resize((dimtabx,dimtaby),PIL.Image.ANTIALIAS)
#                        del orig
#                        ncr.save(lungcoref)        
#        print imgc  
        endnumslice=img.find('.bmp')
        imgcore=img[0:endnumslice]
#        print 'imgcore',imgcore
        posend=endnumslice
        while img.find('-',posend)==-1:
            posend-=1
        debnumslice=posend+1
        slicenumber=int((img[debnumslice:endnumslice])) 
        imscan = Image.open(imgc)
        imscanc= imscan.convert('RGB')
        tablscan = np.array(imscanc)
        if imscan.size[0]>512:
            ncr=imscanc.resize((dimtabx,dimtaby),PIL.Image.ANTIALIAS)
            tablscan = np.array(ncr) 
        ill = 0
      
        foundp=False
        for ll in listnamepatch:
#            print ('1',ll)
            #we read patches in predict/ setnumber and found localisation    
            debsl=ll.find('_')+1
            endsl=ll.find('_',debsl)
            slicename=int(ll[debsl:endsl])
            debx=ll.find('_',endsl)+1
            endx=ll.find('_',debx)
            xpat=int(ll[debx:endx])
            deby=ll.find('_',endx)+1
            endy=ll.find('.',deby)
            ypat=int(ll[deby:endy])

         
        #we find max proba from prediction
            proba=preprob[ill]
           
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
            classlabel=fidclass(prec)
            classcolor=classifc[classlabel]


            if  (slicenumber == slicename )  and (classlabel not in excluvisu):
#                    print(setname, slicename,xpat,ypat,classlabel,classcolor,mproba)
#                    print(mproba,preclass[ill],preprob[ill])
#                    if slicenumber ==2:
#                        print classlabel
#                        print proba
                    foundp=True
                    if classlabel in listlabel:
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                        cur=listlabelaverage[classlabel]
#                               print (numl,cur)
                        averageproba= round((cur*numl+mproba)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabel[classlabel]=1
                        listlabelaverage[classlabel]=mproba
                        
                    if classlabel in listlabelf:
                        nlt=listlabelf[classlabel]
                        listlabelf[classlabel]=nlt+1
#
                    else:
                        listlabelf[classlabel]=1
                    imgi=addpatch(classcolor,classlabel,xpat,ypat)

                    imgt=cv2.add(imgt,imgi)

            ill+=1
#        tabsif=andmerg(tabsim1,tabsi) 
# calculmate contours of patches
        vis=drawContour(imgt,listlabel)
#        print tablscan.shape
#put to zero the contour in image in order to get full visibility of contours
        img2gray = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(tablscan,tablscan,mask = mask_inv)  
#superimpose scan and contours           
        imn=cv2.add(img1_bg,vis)
        
#        imn=cv2.add(tablscan,vis)

        imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
        imgcorefull=imgcore+'.bmp'
        imgname=os.path.join(predictout_dir,imgcorefull)
        cv2.imwrite(imgname,imn)
#        scipy.misc.imsave(imgname, tablscanc)
       
        if foundp:
            t0='average probability'
        else:
            t0='no recognised label'
        t1='n: '+core+' scan: '+str(slicenumber)        
        t2='CONFIDENTIAL - prototype - not for medical use'
        t3='threshold: 0'
        t4=time.asctime()
        tagviews(imgname,t0,0,0,t1,0,20,t2,20,485,t3,0,40,t4,0,50)
        if foundp:
#            tagviews(imgname,'average probability',0,0)           
            for ll in listlabel:
                tagview(imgname,ll,str(listlabelaverage[ll]),175,00)
#        else:   
#            tagviews(imgname,'no recognised label',0,0)
#            errorfile.write('no recognised label in: '+str(f)+' '+str (img)+'\n' )
#            print('no recognised label in: '+str(f)+' '+str (img) )
#    errorfile.write('\n'+'number of labels in :'+str(f)+'\n' )
    for classlabel in listlabelf:  
          print 'patient: ',core,', label:',classlabel,': ',listlabelf[classlabel]
#          string=str(classlabel)+': '+str(listlabelf[classlabel])+'\n' 
##          print string
#          errorfile.write(string )

def renomscan(fa):
#    print(subdir)
        #subdir = top/35
        print('renomscan on:',f)
        num=0
        contenudir = os.listdir(fa)
#        print(contenudir)
        for ff in contenudir:
#            print ff
            if ff.find('.dcm')>0 and ff.find('-')<0:     
                num+=1    
                corfpos=ff.find('.dcm')
                cor=ff[0:corfpos]
                ncff=os.path.join(fa,ff)
#                print ncff
                if num<10:
                    nums='000'+str(num)
                elif num<100:
                    nums='00'+str(num)
                elif num<1000:
                    nums='0'+str(num)
                else:
                    nums=str(num)
                newff=cor+'-'+nums+'.dcm'
    #            print(newff)
                shutil.copyfile(ncff,os.path.join(fa,newff) )
                os.remove(ncff)
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
    pavgene(data)
    dataprocessing(data)
    ILDCNNpredict(data)
    visua(data)

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

predictout_dir = os.path.join(path_patient, predictout)   
remove_folder(predictout_dir)
os.mkdir(predictout_dir)

dirpatientfsdb=os.path.join(path_patient,sroi)


print('work on:',filedcm)
#print filedcm
#namedirtopcf = os.path.join(path_patient,f)
doPrediction(filedcm)
#    print namedirtopcf
        
print('completed on: ',filedcm)
        
t = datetime.datetime.now()
today = str('date: '+dd(t.month)+'-'+dd(t.day)+'-'+str(t.year)+\
'_'+dd(t.hour)+':'+dd(t.minute)+':'+dd(t.second))
print today
#errorfile.close() 
