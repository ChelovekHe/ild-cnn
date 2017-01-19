# coding: utf-8
#sylvain Kritter 19-jan-2017
'''3 dpredict on lung
 '''
import os
import cv2
#import dircache
import sys
import shutil
from scipy import misc
import numpy as np

import cPickle as pickle
import dicom
from Tkinter import *
import scipy
import PIL
import cPickle as pickle

#from PIL import Image as ImagePIL
from PIL import Image,ImageFont, ImageDraw
os.environ['KERAS_BACKEND'] = 'theano'
#from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
import cnn_model as CNN4
import time
from time import time as mytime


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
t0=mytime()
#####################################################################
#define the working directory
globalHist=True 
contrast=False
normiInternal =True
isGre=False
HUG='predict_23d'   
HUG='predict_106530'   
HUG='predict_S14740' 

#HUG='HUG'
#subDir='ILDt'
predictout='predicted_results'
predictout3d='predicted_results_3d'
predictout3dr='predicted_results_3dr'
predictoutmerge='predicted_results_merge'
source_name='source'
jpegpath='jpegpath'
jpegpath3d='jpegpath3d'
jpegpadirm='jpegpadirm'

lung_name='lung'
lung_namebmp='bmp'
lung_name_gen='lung_mask'
scan_bmp='scan_bmp'
transbmp='trans_bmp'
sroid='sroi3d'
sroi='sroi'
bgdir='bgdir3d'

typei='jpg'
typeid='jpg' #can be png for 16b
typej='jpg'
typeiroi='bmp'
excluvisu=['back_ground','healthy']
#excluvisuh=['bronchiectasis']
#excluvisu=[]
excluvisuh=[]



threeFileTxt='uip.txt'
threeFile='uip.html'
threeFilejs='world.js'

threeFileTxtMerge='uipMerge.txt'
threeFileMerge='uipMerge.html'
threeFilejsMerge='worldMerge.js'

threeFileTxt3d='uip3d.txt'
threeFile3d='uip3d.html'
threeFilejs3d='world3d.js'

threeFileTop='uiptop.html'
threeFileBot='uipbot.html'


dimpavx=16
dimpavy=16
pxy=float(dimpavx*dimpavy)
#imageDepth=65535 #number of bits used on dicom images (2 **n) 13 bits
imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
#imageDepth=255 #number of bits used on dicom images (2 **n) 13 bits
#patch overlapp tolerance
thrpatch = 0.99 #threshold for pad acceptance (recovery)
thrproba =0.5 #thresholm proba for generation of predicted images
thrprobaUIP=0.6 #threshold proba for UIP

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)


listHug= [ name for name in os.listdir(dirHUG) if os.path.isdir(os.path.join(dirHUG, name)) ]

print listHug


font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)
#create patch and jpeg directory

#remove_folder(jpegpath)

eferror=os.path.join(dirHUG,'genepatcherrortop3d.txt')

    
    
picklein_file = '../pickle_ex/pickle_ex53'
picklein_file_front = '../pickle_ex/pickle_ex59'
modelname='ILD_CNN_model.h5'
avgPixelSpacing=0.734   # average pixel spacing
###############################################################
classif ={
            'back_ground':0,
            'consolidation':1,
            'HC':2,
            'ground_glass':3,
            'healthy':4,
            'micronodules':5,
            'reticulation':6,
            'air_trapping':7,
            'cysts':8,
            'bronchiectasis':9,
            'bronchial_wall_thickening':10,
             'early_fibrosis':11,
             'emphysema':12,
             'increased_attenuation':13,
             'macronodules':14,
             'pcp':15,
             'peripheral_micronodules':16,
             'tuberculosis':17      
            }

clssifcount={
            'back_ground':0,
            'consolidation':0,
            'HC':0,
            'ground_glass':0,
            'healthy':0,
            'micronodules':0,
            'reticulation':0,
            'air_trapping':0,
            'cysts':0,
            'bronchiectasis':0,
            'bronchial_wall_thickening':0,
             'early_fibrosis':0,
             'emphysema':0,
             'increased_attenuation':0,
             'macronodules':0,
             'pcp':0,
             'peripheral_micronodules':0,
             'tuberculosis':0      
            }
usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'emphysema'
        ]
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,255)
lightgreen=(125,237,125)
orange=(255,153,102)
lowgreen=(0,51,51)

volcol={          
'back_ground':'boxMaterialGrey',
#'consolidation':'boxMaterialRed',
'consolidation':'boxMaterialCyan',
'HC':'boxMaterialBlue',
#'ground_glass':'boxMaterialYellow',
'ground_glass':'boxMaterialRed',
'healthy':'boxMaterialGrey',
'micronodules':'boxMaterialGreen',
#'reticulation':'boxMaterialPurple',
'reticulation':'boxMaterialYellow',
'air_trapping':'boxMaterialPink',
'cysts':'boxMaterialLightgreen',
 'bronchiectasis':'boxMaterialOrange',
 'bronchial_wall_thickening':'boxMaterialWhite',
 'early_fibrosis':'boxMaterialWhite',
 'emphysema':'boxMaterialWhite',
 'increased_attenuation':'boxMaterialWhite',
 'macronodules':'boxMaterialWhite',
 'pcp':'boxMaterialWhite',
 'peripheral_micronodules':'boxMaterialWhite',
 'tuberculosis':'boxMaterialWhite'}  
          
classifc ={
'back_ground':darkgreen,
#'consolidation':red,
'consolidation':cyan,
'HC':blue,
'ground_glass':red,
#'ground_glass':yellow,
'healthy':darkgreen,
#'micronodules':cyan,
'micronodules':green,
#'reticulation':purple,
'reticulation':yellow,
'air_trapping':pink,
'cysts':lightgreen,
 'bronchiectasis':orange,
 'nolung': lowgreen,
 'bronchial_wall_thickening':white,
 'early_fibrosis':white,
 'emphysema':white,
 'increased_attenuation':white,
 'macronodules':white,
 'pcp':white,
 'peripheral_micronodules':white,
 'tuberculosis':white
 }
#pickle_dir=os.path.join(cwdtop,pickel_dirsource) 
#pickle_dirToMerge=os.path.join(cwdtop,pickel_dirsourceToMerge)  

#patch_dir=os.path.join(dirHUG,patch_dirsource)
#print patch_dir
def rsliceNum(s,c,e):
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 

def normi(tabi):
     """ normalise patches"""
    
     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))
    
     mm=max_val-min_val
     mm=max(mm,1.0)
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(imageDepth/mm)
     tabi2=tabi2.astype('uint16')

     return tabi2
     
def reshapeScan(tabscan,slnt,dimtabx):
    print 'reshape'
    tabres = np.zeros((dimtabx,slnt,dimtabx), np.uint16)
    for i in range (0,dimtabx):
        for j in range (0,slnt):
            tabres[i][j]=tabscan[j][i]
    return tabres
        
#remove_folder(pickle_dir)
#os.mkdir(pickle_dir)  
def genebmp(fn):
    """generate patches from dicom files"""
    
    print ('load dicom files in :',fn)
    (top,tail) =os.path.split(fn)
#    sroidir=os.path.join(top,sroi)
    #directory for patches
    fmbmp=os.path.join(fn,scan_bmp)
    remove_folder(fmbmp)
    os.mkdir(fmbmp)
    listdcm=[name for name in  os.listdir(fn) if name.lower().find('.dcm')>0]
   
    FilesDCM =(os.path.join(fn,listdcm[0]))  
    #           
    RefDs = dicom.read_file(FilesDCM)
    #    print RefDs
    SliceThickness=RefDs.SliceThickness
    try:
            SliceSpacingB=RefDs. SpacingBetweenSlices
    except AttributeError:
             print "Oops! No Slice spacing..."
             SliceSpacingB=0
    
    slicepitch=float(SliceThickness)+float(SliceSpacingB)
    print SliceThickness
    print SliceSpacingB
    print 'slice pitch in z :',slicepitch
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing 
    dsr= RefDs.pixel_array
    imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=imgresize.shape[0]
    dimtaby=imgresize.shape[1]
    print dimtabx, dimtaby
    slnt=0
    for l in listdcm:
        
        FilesDCM =(os.path.join(fn,l))  
        RefDs = dicom.read_file(FilesDCM)
        slicenumber=int(RefDs.InstanceNumber)
        if slicenumber> slnt:
            slnt=slicenumber
    slnt=slnt+1
    print 'number of slices', slnt-1
    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.uint16)
    for l in listdcm:
#        print l
        FilesDCM =(os.path.join(fn,l))  
    #           
        RefDs = dicom.read_file(FilesDCM)
  
        dsr= RefDs.pixel_array
        dsr= dsr-dsr.min()
        dsr=dsr.astype('uint16')
           
        slicenumber=int(RefDs.InstanceNumber)
        endnumslice=l.find('.dcm') 
        imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)

        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
    #            print 'imgcore' , imgcore
        bmpfile=os.path.join(fmbmp,imgcoreScan)
        scipy.misc.imsave(bmpfile,imgresize)
        if globalHist ==True :
                imgresize = normi(imgresize) 
#                print imgresize.min(),imgresize.max()
#            if tail =='lung':
#                imgresize = normi(imgresize) 
        tabscan[slicenumber]=imgresize
    return tabscan,slnt,dimtabx,slicepitch
    
def genebmplung(fn,lungname,slnt,dimtabx,dimtaby):
    """generate patches from dicom files"""

    print ('load dicom files for lung in :',fn)
    (top,tail) =os.path.split(fn)
   
    fmbmp=os.path.join(fn,lungname)
    fmbmpbmp=os.path.join(fmbmp,lung_namebmp)
#    print fmbmpbmp
    remove_folder(fmbmpbmp)
    
    os.mkdir(fmbmpbmp)
#    print 'aa'
    listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]
   
    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
    for l in listdcm:
#        print l
        FilesDCM =(os.path.join(fmbmp,l))  
    #           
        RefDs = dicom.read_file(FilesDCM)
  
        dsr= RefDs.pixel_array
        dsr= dsr-dsr.min()
        c=float(255)/dsr.max()
        dsr=dsr*c     
        dsr=dsr.astype('uint8')
    
        fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing 
        imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
##        min_val=np.min(dsr)
         
           
        slicenumber=int(RefDs.InstanceNumber)
        endnumslice=l.find('.dcm') 

    #        cv2.imwrite(bmpfile, imgresize)
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
    #            print 'imgcore' , imgcore
        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        scipy.misc.imsave(bmpfile,imgresize)
#            if tail =='lung':
#                imgresize = normi(imgresize) 
        tabscan[slicenumber]=imgresize
    return tabscan 

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m
    
def fidclass(numero,classn):
    """return class from number"""
    found=False
#    print numero
    for cle, valeur in classn.items():
        
        if valeur == numero:
            found=True
            return cle     
    if not found:
        return 'unknown'  
        
#        print proba[0]
def contour2(im,l,dimtabx,dimtaby):  
    col=classifc[l]
#    print l
#    print 'dimtabx' , dimtabx
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis


def tagview(fig,label,x,y):
    """write text in image according to label and color"""
#    print ('write label :',label,' at: ', fig)
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=classifc[label]
    labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
#        deltax=0
        deltay=60
    else:        
        deltay=25*((labnow-1)%5)
#        deltax=175*((labnow-1)//5)
#        deltax=80*((labnow-1)//5)

#    print (x+deltax,y+deltay)
    draw.text((x, y+deltay),label,col,font=font10)
    imgn.save(fig)

def tagviewt(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font10)
    imgn.save(fig)

    
def tagviews(b,fig,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    if b:
        draw.rectangle ([x1, y1,x1+100, y1+15],outline='black',fill='black')
        draw.rectangle ([140, 0,dimtabx,75],outline='black',fill='black')
    draw.text((x0, y0),t0,white,font=font10)
    draw.text((x1, y1),t1,white,font=font10)
    draw.text((x2, y2),t2,white,font=font10)
    if not b:
        draw.text((x3, y3),t3,white,font=font10)
    draw.text((x4, y4),t4,white,font=font10)
    draw.text((x5, y5),t5,white,font=font10)
    imgn.save(fig)
 

def pavgene(dirf,dimtabx,dimtaby,tabscanScan,tabscanLung,slnt,jpegpath):
        """ generate patches from scan"""
        patch_list=[]   
        (dptop,dptail)=os.path.split(dirf)
        print('generate patches on: ',dptail)
        jpegpathdir=os.path.join(dirf,jpegpath)
        remove_folder(jpegpathdir)
        os.mkdir(jpegpathdir)        
        for img in range (0,slnt):

             tabfw = np.zeros((dimtabx,dimtaby,3), np.uint8)
             tablung = np.copy(tabscanLung[img])
             tabfrgb=np.copy(tablung)
             np.putmask(tablung,tablung>0,1)
             np.putmask(tabfrgb,tabfrgb>0,100)
             tabfrgb= cv2.cvtColor(tabfrgb,cv2.COLOR_GRAY2BGR)
             tabf=tabscanScan[img]
             nz= np.count_nonzero(tablung)
             if nz>0:
            
                 atabf = np.nonzero(tablung)
                #tab[y][x]  convention
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 ymin=atabf[0].min()
                 ymax=atabf[0].max()
                
                 i=xmin
                 while i < xmax:
                     j=ymin
                     while j<ymax:
                         
                         tabpatch=tablung[j:j+dimpavy,i:i+dimpavx]
                         area= tabpatch.sum()  
                         targ=float(area)/pxy
                             
                         if targ>=thrpatch:
    #                        print i,j,targ,area
    #                        ooo
                            imgray = tabf[j:j+dimpavy,i:i+dimpavx]
                            
                            imagemax= cv2.countNonZero(imgray)
                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)

                            if imagemax > 0 and min_val != max_val:
    #                            namepatch=patchpathf+'/p_'+str(slicenumber)+'_'+str(i)+'_'+str(j)+'.'+typei
                                if contrast:
                                        if normiInternal:    
                                            tabi2=normi(imgray)
     
                                        else:
                                            if imageDepth<256:
                                                tabi2 = cv2.equalizeHist(imgray)     
                                            else:
    #                                           tabi2=normi(imgray)
                                               tabi2= cv2.normalize(imgray, 0, imageDepth,cv2.NORM_MINMAX);
                                        patch_list.append((img,i,j,tabi2))
                                else:
                                    patch_list.append((img,i,j,imgray))

                                tablung[j:j+dimpavy,i:i+dimpavx]=0
                                x=0
                                while x < dimpavx:                                   
                                    y=0
                                    while y < dimpavy:
                                        if y+j<dimtabx and x+i<dimtaby:
                                            tabfw[y+j][x+i]=[125,0,0]    
                                        if x == 0 or x == dimpavx-1 :
                                            y+=1
                                        else:
                                            y+=dimpavy-1
                                    x+=1                                             
                                j+=dimpavy-1
                         j+=1
                     i+=1

                 nameslijpeg='s_'+str(img)+'.'+typej
                 namepatchImage=os.path.join(jpegpathdir,nameslijpeg)
                 tabjpeg=cv2.add(tabfw,tabfrgb)
                 scipy.misc.imsave(namepatchImage,tabjpeg)

        return patch_list
        
def ILDCNNpredict(patch_list,model):
    print ('Predict started ....')
    dataset_list=[]
    for fil in patch_list:             
              dataset_list.append(fil[3])        
    X0=len(dataset_list) 
    # adding a singleton dimension and rescale to [0,1]
    pa = np.asarray(np.expand_dims(dataset_list, 1)) / float(imageDepth)
    # look if the predict source is empty
    # predict and store  classification and probabilities if not empty
    if X0 > 0:
        proba = model.predict_proba(pa, batch_size=100)

    else:
        print (' no patch in selected slice')
        proba = ()
    print 'number of patches', len(pa)

    return proba

def wtebres(dirf,tab,dimtabx,slicepitch,lungm,ty):
    print 'generate front images from',ty
    (top,tail)=os.path.split(dirf)
    bgdirf=os.path.join(dirf,lungm)
    bgdirf=os.path.join(bgdirf,transbmp)
    remove_folder(bgdirf)
    os.mkdir(bgdirf)
    fxs=float(slicepitch/avgPixelSpacing )
#    print tab[0].shape[0]
    ntd=int(round(fxs*tab[0].shape[0],0))
    if ty=='scan':
        tabres=np.zeros((dimtabx,ntd,dimtabx),np.uint16)
    else:
        tabres=np.zeros((dimtabx,ntd,dimtabx),np.uint8)
    for i in range (0,dimtabx):
#        print i, tab[i].max()
        if tab[i].max()>0:
#            print tab[i].max()
#            print tab[i].shape
            
            imgresize=cv2.resize(tab[i],None,fx=1,fy=fxs,interpolation=cv2.INTER_LINEAR)
#            print i, imgresize.min(), imgresize.max()
#            print dimtabx,fxs,imgresize.shape,tab[i].shape

            if ty=='scan':
                typext=typeid
            else:
                typext=typei
            trscan='tr_'+str(i)+'.'+typext
            trscanbmp='tr_'+str(i)+'.'+typei
            if ty=='lung':
                namescan=os.path.join(bgdirf,trscanbmp)
                np.putmask(imgresize,imgresize>0,100)
                scipy.misc.imsave(namescan,imgresize) 
            if ty=='scan':
                namescan=os.path.join(wridir,trscan)

                scipy.misc.imsave(namescan,imgresize)
               
            dimtabxn=imgresize.shape[0]
            dimtabyn=imgresize.shape[1]
#            print imgresize.shape,tabres.shape
            tabres[i]=imgresize
#        cv2.imwrite (trscan, tab[i],[int(cv2.IMWRITE_PNG_COMPRESSION),0])
    return dimtabxn,dimtabyn,tabres


def modelCompilation(t):

    if t=='cross':
        modelpath = os.path.join(picklein_file, 'ILD_CNN_model.h5')
    if t=='front':
        modelpath = os.path.join(picklein_file_front, 'ILD_CNN_model.h5')

    if os.path.exists(modelpath):
        model = load_model(modelpath)
        model.compile(optimizer='Adam', loss=CNN4.get_Obj('ce'))
        return model
        
def addpatch(col,lab, xt,yt,px,py,dimtabx,dimtaby):
    imgi = np.zeros((dimtabx,dimtaby,3), np.uint8)
    tablint=[(xt,yt),(xt,yt+py),(xt+px,yt+py),(xt+px,yt)]
    tabtxt=np.asarray(tablint)
    cv2.polylines(imgi,[tabtxt],True,col)
    cv2.fillPoly(imgi,[tabtxt],col)
    return imgi        
        
def drawContour(imi,ll,dimtabx,dimtaby):    
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

def tagviewn(fig,label,pro,nbr,x,y):
    """write text in image according to label and color"""
    col=classifc[label]
#    print col, label    
    labnow=classif[label]-1  
    if label == 'back_ground':
        x=0
        y=0        
        deltax=0
        deltay=60
    else:                
        deltax=130*((labnow)//5)
        deltay=11*((labnow)%5)
    gro=-x*0.0027+1.2
#    if label=='ground_glass':
#            print label,x,deltax, y,deltay,labnow,gro
    cv2.putText(fig,str(nbr)+' '+label+' '+pro,(x+deltax, y+deltay+10),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

      
def  visua(dirf,probaInput,patch_list,dimtabx,dimtaby,slnt,predictout,sroi,scan_bmp):

    print('visualisation',scan_bmp)
    sliceok=[]
    (dptop,dptail)=os.path.split(dirf)
    predictout_dir = os.path.join(dirf, predictout)
#    print predictout_dir
    remove_folder(predictout_dir)
    os.mkdir(predictout_dir)      
    for i in range (0,len(classif)):
#        print 'visua dptail', topdir
        listelabelfinal[fidclass(i,classif)]=0
    #directory name with predict out dabasase, will be created in current directory
    preprob=probaInput
 
    dirpatientfdb=os.path.join(dirf,scan_bmp)
    dirpatientfsdb=os.path.join(dirf,sroi)
    listbmpscan=os.listdir(dirpatientfdb)
#    print dirpatientfdb
    listlabelf={}
    sroiE=False
    if os.path.exists(dirpatientfsdb):
        sroiE=True
        listbmpsroi=os.listdir(dirpatientfsdb)
  
    for img in listbmpscan:
        slicenumber= rsliceNum(img,'_','.'+typei)
        imgt = np.zeros((dimtabx,dimtaby,3), np.uint8)
        listlabelaverage={}
        listlabel={}
        listlabelrec={}
#        print dirpatientfsdb,dirpatientfdb
        if sroiE:
            for imgsroi in listbmpsroi:
#                print imgsroi
                slicenumbersroi=rsliceNum(imgsroi,'_','.'+typeiroi)
                if slicenumbersroi==slicenumber:
                    imgc=os.path.join(dirpatientfsdb,imgsroi)
                    break
        else:
            imgc=os.path.join(dirpatientfdb,img)
        tablscan=cv2.imread(imgc,1)
#        ill = 0
        foundp=False
        for ll in range(0,len(patch_list)):
         
            slicename=patch_list[ll][0] 
            xpat=patch_list[ll][1]
            ypat=patch_list[ll][2]
            proba=preprob[ll]          
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
            classlabel=fidclass(prec,classif) 
            classcolor=classifc[classlabel]      
#            print xpat,ypat,mprobai,slicename
            if slicenumber == slicename and classlabel not in excluvisu:
                    foundp=True
                    if classlabel in listlabel:
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                    else:
                        listlabel[classlabel]=1
                    if classlabel in listlabelf:
                        nlt=listlabelf[classlabel]
                        listlabelf[classlabel]=nlt+1
                    else:
                        listlabelf[classlabel]=1     
            if mprobai >=thrproba and slicenumber == slicename and classlabel not in excluvisu:
                    if slicenumber not in sliceok:
                        sliceok.append(slicenumber)
                    if classlabel in listlabelrec:
                        numl=listlabelrec[classlabel]
                        listlabelrec[classlabel]=numl+1
                        cur=listlabelaverage[classlabel]
                        averageproba= round((cur*numl+mproba)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabelrec[classlabel]=1
                        listlabelaverage[classlabel]=mproba

                    imgi=addpatch(classcolor,classlabel,xpat,ypat,dimpavx,dimpavy,dimtabx,dimtaby)
                    imgt=cv2.add(imgt,imgi)
                    
#            ill+=1
        tablscan = cv2.cvtColor(tablscan, cv2.COLOR_BGR2RGB)

        vis=drawContour(imgt,listlabel,dimtabx,dimtaby)

        imn=cv2.add(tablscan,vis)

        if foundp:
#            tagviews(imgname,'average probability',0,0)           
                for ll in listlabelrec:
                    delx=int(dimtaby*0.6-120)
                    tagviewn(imn,ll,str(listlabelaverage[ll]),listlabelrec[ll],delx,0)
        else:   
#            tagviews(imgname,'no recognised label',0,0)
                errorfile.write('no recognised label in: '+str(dptail)+' '+str (img)+'\n' )

        imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)

        predict_outFile=os.path.join( predictout_dir,img)
        cv2.imwrite(predict_outFile,imn)            
            
       
        if foundp:
                t0='average probability'
        else:
                t0='no recognised label'
        t1='n: '+dptail+' scan: '+str(slicenumber)        
        t2='CONFIDENTIAL - prototype - not for medical use'
        t3='For threshold: '+str(thrproba)+' :'
        t4=time.asctime()
        (topw,tailw)=os.path.split(picklein_file)
        t5= tailw
        tagviews(False,predict_outFile,t0,0,0,t1,0,20,t2,20,dimtaby-27,t3,0,38,t4,0,dimtaby-10,t5,0,dimtaby-20)        
        errorfile.write('\n'+'number of labels in :'+str(dptop)+' '+str(dptail)+str (img)+'\n' )
#    print listlabelf
    for classlabel in listlabelf:  
          listelabelfinal[classlabel]=listlabelf[classlabel]
          print 'patient: ',dptail,', label:',classlabel,': ',listlabelf[classlabel]
          stringtw=str(classlabel)+': '+str(listlabelf[classlabel])+'\n' 
#          print string
          errorfile.write(stringtw )
    return sliceok

def genethreef(dirpatientdb,patchPositions,probabilities_raw,slicepitch,dimtabx,dimtaby,dimpavx,lsn,v):
        """generate  voxels for 3d view"""
        (dptop,dptail)=os.path.split(dirpatientdb)
        pz=slicepitch/avgPixelSpacing
        cwd=os.getcwd()
        souuip=os.path.join(cwd,threeFileTop)
#        print souuip
        if v =='cross':
            threeFilel=dptail+'_'+threeFile
            threeFileTxtl=dptail+'_'+threeFileTxt
            jsfile=dptail+'_'+threeFilejs
            BGx=str(dimpavx)
            BGy=str(dimpavx)
            BGz=str(round(pz,3))
            
        if v =='merge':
            threeFilel=dptail+'_'+threeFileMerge
            threeFileTxtl=dptail+'_'+threeFileTxtMerge
            jsfile=dptail+'_'+threeFilejsMerge
            BGx=str(dimpavx)
            BGy=str(dimpavx)
            BGz=str(round(pz,3))
         
            
        if v =='front':
            threeFilel=dptail+'_'+threeFile3d
            threeFileTxtl=dptail+'_'+threeFileTxt3d
            jsfile=dptail+'_'+threeFilejs3d
            BGy=str(dimpavx)
            BGx=str(round(pz,3))
            BGz=str(dimpavx)
    
        desouip=os.path.join(dirpatientdb,threeFilel)
        shutil.copyfile(souuip,desouip)                    
        volumefileT = open(os.path.join(dirpatientdb,threeFileTxtl), 'w')                  
        jsfilel=os.path.join(dirpatientdb,jsfile)
        volumefile = open(os.path.join(dirpatientdb,threeFilel), 'a')
        volumefilejs = open(jsfilel, 'w')
#        jsfilel.replace("\\","/")
#        print jsfilel
        volumefile.write( '<script src="'+jsfilel+'"></script>\n)')	
        volumefilejs.write( 'function buildobj()	{\n')	

#        zxd=slicepitch*lsn/2/avgPixelSpacing
        zxd=pz*(lsn-1)/2
        volumefileT.write('camera.position.set(0 '+', -'+BGx+', 0 );\n')  
        
        volumefileT.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
        BGy+' , '+BGz+' ) ;\n\n')
        volumefileT.write('var voxels = [\n')
        volumefilejs.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
        BGy+' , '+BGz+' ) ;\n')

        print 'pz',pz
     
#        volumefile.write('camera.position.set('+str(dimtabx/2)+',-'+str(dimtabx/2)+','+str(zxd)+' );\n') 
#        volumefile.write('controls.target.set('+str(dimtabx/2)+','+str(dimtabx/2)+','+str(zxd)+') ;\n') 
        volumefilejs.write('camera.position.set(0 '+', -'+str(dimtaby)+', 0 );\n') 
        volumefilejs.write('controls.target.set(0 , 0 , 0 );\n') 
        
#    print  >> volumefile ,dictSurf
#    volumefile.close()
#    print dictSurf
        
        
        for ll in range(0,len(patchPositions)):
            
            slicename=patchPositions[ll][0] 
#            print slicename
            if v =='cross' or v=='merge':
                 xpat=dimtaby/2-patchPositions[ll][1]
                 ypat=patchPositions[ll][2]-(dimtabx/2)
                 zpa=round((lsn-slicename)*pz-zxd,3)
            
                
            if v =='front':
               
                ypat=-round((lsn-slicename)*pz-zxd,3)
                xpat=(dimtaby/2)-patchPositions[ll][1]
                zpa=(dimtabx/2)-patchPositions[ll][2]    
                
            
            proba=probabilities_raw[ll]          
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
#            print '1', mproba
#            print classif[nset],nset,prec
            classlabel=fidclass(prec,classif) 

            bm=volcol[classlabel]            
            bmn = 'boxMesh'+str(ll)
            
#                print (slicename,zpa)
            if ll ==0:
                volumefileT.write('{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' },\n')
            elif ll==1:
                 volumefileT.write('{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' }')
            else:                     
                 volumefileT.write(',\n{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' }')

            volumefilejs.write( 'var newMaterial= '+bm+'.clone();\n'); 
            if mproba>= thrprobaUIP:
#                print '2', mproba
#                mproba=(mproba-thrprobaUIP)/thrprobaUIP
                volumefilejs.write( 'newMaterial.opacity= '+str(mproba)+';\n'); 
            else:
                volumefilejs.write( 'newMaterial.opacity= 0;\n'); 
            volumefilejs.write( bmn+' = new THREE.Mesh(boxGeometry,newMaterial)\n'); 
            volumefilejs.write( bmn+'.position.set('+str(xpat)+', '+str(ypat)+','+str(zpa)+');\n')
            volumefilejs.write('scene.add('+bmn+');\n') 

        volumefileT.write('\n]\n')
        vtb=open(os.path.join(cwd,threeFileBot),'r')
        app=vtb.read()
        volumefile.write(app)
        volumefilejs.write('\n}\n')
        volumefile.close()
        volumefileT.close()
        vtb.close()
        volumefilejs.close()

def genecross(proba_front,patch_list_front,slnt,slicepitch,dimtabx,dimtaby):
    proba_cross_gene=[]
    patch_list_cross_gene=[]
    slicewidth={}
#    tab=np.zeros((dimtabx,dimtaby),np.uint8)
    for i in range (1, slnt):
        imin=(i-1)*slicepitch
        imax=i*slicepitch
        slicewidth[i]=(imin,imax)
    for i in range (1, slnt):    
        for ll in range(0,len(patch_list_front)):             
            ypat=patch_list_front[ll][2]*avgPixelSpacing
            if ypat>= slicewidth[i][0] and ypat< slicewidth[i][1]:
                proba=proba_front[ll]          
                prec, mprobai = maxproba(proba)
                if mprobai>thrprobaUIP:
                    patch_list_cross_gene.append((i,patch_list_front[ll][1],patch_list_front[ll][0]))
                    proba_cross_gene.append(proba_front[ll])
#                  print i, ll[1],ll[2],ypat
#                  tab[ll[2]:ll[2]+dimpavy,ll[1]:ll[1]+dimpavx]=100
#                  cv2.imshow('tab',tab)
#                  cv2.waitKey(0)    
#                  cv2.destroyAllWindows()
#                  ooo
    return proba_cross_gene,patch_list_cross_gene

def createProba(pat,pr,px):
    n=classif[pat]
    proba=[]
    l=len(px)
#    print n ,l
    for i in range(0,l):
        if i==n:
            proba.append(pr)
        else:
            proba.append(0)    
    return proba

    
def pavf(pat,scan,tab,dimpavx,dimpavy,dimtabx,dimtaby,jpegpath,patch_list,proba_list): 
#     print 'pav merge', pat,scan
     tabfw=np.zeros((dimtabx,dimtaby,3),np.uint8)
     jpegpathdir=os.path.join(dirf,jpegpath)
     if not os.path.exists(jpegpathdir):
         os.mkdir(jpegpathdir)   
     tabbc=np.copy(tab)
     tabr=np.copy(tab)
     tabr= cv2.cvtColor(tabr,cv2.COLOR_GRAY2BGR)
     atabf = np.nonzero(tab)
     pxy=float(dimpavx*dimpavy)
     np.putmask(tabbc,tabbc>0,1)
                #tab[y][x]  convention
     xmin=atabf[1].min()
     xmax=atabf[1].max()
     ymin=atabf[0].min()
     ymax=atabf[0].max()
    
     i=xmin
     while i < xmax:
         j=ymin
         while j<ymax:
             
             tabpatch=tabbc[j:j+dimpavy,i:i+dimpavx]
             area= tabpatch.sum()  
             targ=float(area)/pxy
                 
             if targ>thrpatch:
#                print i,j,targ,area
                patch_list.append((scan,i,j))
                proba_list.append(createProba(pat,thrprobaUIP,proba_cross[0]))

                tabbc[j:j+dimpavy,i:i+dimpavx]=0
                x=0
                while x < dimpavx:                                   
                        y=0
                        while y < dimpavy:
                            if y+j<dimtabx and x+i<dimtaby:
                                tabfw[y+j][x+i]=[125,0,0]    
                            if x == 0 or x == dimpavx-1 :
                                y+=1
                            else:
                                y+=dimpavy-1
                        x+=1                                             
                j+=dimpavy-1
             j+=1
         i+=1

     nameslijpeg=pat+'s_'+str(scan)+'.'+typej
     namepatchImage=os.path.join(jpegpathdir,nameslijpeg)
     if os.path.exists(namepatchImage):
         tabr=cv2.imread(namepatchImage,1)
#     print tabfw.shape,tabr.shape
     tabjpeg=cv2.add(tabfw,tabr)
#     cv2.imshow('tabfw'+pat+str(scan),tabfw)
#     cv2.imshow('tabjpeg',tabjpeg)
#     cv2.imshow('tabr',tabr)
#     cv2.waitKey(0)    
#     cv2.destroyAllWindows()
     
     scipy.misc.imsave(namepatchImage,tabjpeg)
     return proba_list,patch_list
     
def mergeproba(prx,plx,prg,plg,slnt,dimtabx,dimtaby)  :  
    print "merge proba list"
    
    patch_list_merge=[]
    proba_merge=[]
    plgd={}
    plxd={}
    prgd={}
    prxd={}
    
    listp=[name for name in usedclassif if name not in excluvisu]  
   
    for i in usedclassif:
        plgd[i]=[]
        plxd[i]=[]
        prgd[i]=[]
        prxd[i]=[]
    probabg=createProba('back_ground',thrprobaUIP,proba_cross[0])
    print 'sort plg'
    for ll in range(0,len(plg)):  
        proba=prg[ll]          
        prec, mprobai = maxproba(proba)
        classlabel=fidclass(prec,classif) 
        if mprobai> thrprobaUIP:
            plgd[classlabel].append(plg[ll])
            prgd[classlabel].append(prg[ll])
        else:
            plgd['back_ground'].append(plg[ll])
            prgd['back_ground'].append(probabg)
   
    print 'sort plx'    
    for ll in range(0,len(plx)):  
        proba=prx[ll]          
        prec, mprobai = maxproba(proba)
        classlabel=fidclass(prec,classif) 
        if mprobai> thrprobaUIP:
            plxd[classlabel].append(plx[ll])
            prxd[classlabel].append(prx[ll])
        else:
            plxd['back_ground'].append(plx[ll])
            prxd['back_ground'].append(probabg)
   
    
    print 'merge pattern'
    for p in listp:
#        print p
        for i in range (1, slnt): 
            tab0=np.zeros((dimtabx,dimtaby),np.uint8)
            tab1=np.zeros((dimtabx,dimtaby),np.uint8)
            tab2=np.zeros((dimtabx,dimtaby),np.uint8)
            for ll in range(0,len(plgd[p])):
                if plgd[p][ll][0]==i:                    
                    tab0[plgd[p][ll][2]:plgd[p][ll][2]+dimpavy,plgd[p][ll][1]:plgd[p][ll][1]+dimpavx]=100          
            for ll in range(0,len(plxd[p])):                
                if plxd[p][ll][0]==i:
                    tab1[plxd[p][ll][2]:plxd[p][ll][2]+dimpavy,plxd[p][ll][1]:plxd[p][ll][1]+dimpavx]=100
            tab2=np.bitwise_and(tab0,tab1)
            nz= np.count_nonzero(tab2)
            if nz>0:
                proba_merge,patch_list_merge=pavf(p,i,tab2,dimpavx,dimpavy,dimtabx,dimtaby,jpegpadirm,patch_list_merge,proba_merge)
    print 'merge exclu'
    for p in excluvisu:
#        print p
        for i in range (1, slnt): 
            tab2=np.zeros((dimtabx,dimtaby),np.uint8)
            for ll in range(0,len(plxd[p])):                
                if plxd[p][ll][0]==i:
                    tab2[plxd[p][ll][2]:plxd[p][ll][2]+dimpavy,plxd[p][ll][1]:plxd[p][ll][1]+dimpavx]=100
            nz= np.count_nonzero(tab2)
            if nz>0:
                proba_merge,patch_list_merge=pavf(p,i,tab2,dimpavx,dimpavy,dimtabx,dimtaby,jpegpadirm,patch_list_merge,proba_merge)
#                if len(pr)>0:
#                    proba_merge.append(pr)
#                    patch_list_merge.append(pl)
#    print patch_list_merge
    return proba_merge,patch_list_merge
     

         
for f in listHug:
    print f
 
    errorfile = open(eferror, 'a')
    listelabelfinal={}
    dirf=os.path.join(dirHUG,f)
    sroidir=os.path.join(dirf,sroid)
    wridir=os.path.join(dirf,transbmp)
    remove_folder(wridir)
    os.mkdir(wridir)
#    bgdirf=os.path.join(dirf,bgdir)
#    remove_folder(bgdirf)
#    os.mkdir(bgdirf)   
#    listsliceok=[]
    if isGre:
        dirg=os.path.join(dirf,source_name)
        tabscan,slnt,dimtabx,slicepitch=genebmp(dirg)
        tabres=reshapeScan(tabscan,slnt,dimtabx)
        dimtabx,dimtaby=wtebres(dirg,tabres,dimtabx,slicepitch)

        dirg=os.path.join(dirf,lung_name)
        tabscan,slnt,dimtabx,slicepitch=genebmp(dirg)
        tabres=reshapeScan(tabscan,slnt,dimtabx)
        dimtabx,dimtaby=wtebres(dirg,tabres,dimtabx,slicepitch)
        listdirf=[name for name in os.listdir(dirf) if name in usedclassif]
        print listdirf
        for g in listdirf:           
            
            dirg=os.path.join(dirf,g)
            print g
            
            tabscan,slnt,dimtabx,slicepitch=genebmp(dirg)
            tabres=reshapeScan(tabscan,slnt,dimtabx,slicepitch)
            dimtabx,dimtaby=wtebres(dirg,tabres,dimtabx,slicepitch)

    else:
        
        tabscanScan,slnt,dimtabx,slicepitch=genebmp(dirf)
        ##slnt= number of slices +1
#        print 'dirf', dirf
        tabscanLung=genebmplung(dirf,lung_name_gen,slnt,dimtabx,dimtabx)

        patch_list=pavgene(dirf,dimtabx,dimtabx,tabscanScan,tabscanLung,slnt,jpegpath)
        model=modelCompilation('cross')
        proba=ILDCNNpredict(patch_list,model)
        pickle.dump(proba, open( os.path.join(dirf,"proba_cross"), "wb" ))
        pickle.dump(patch_list, open( os.path.join(dirf,"patch_list_cross"), "wb" ))
        sliceok = visua(dirf,proba,patch_list,dimtabx,dimtabx,slnt,predictout,sroi,scan_bmp)
        tabresScan=reshapeScan(tabscanScan,slnt,dimtabx)
        dimtabxn,dimtabyn,tabScan3d=wtebres(dirf,tabresScan,dimtabx,slicepitch,lung_name_gen,'scan')
        tabresLung=reshapeScan(tabscanLung,slnt,dimtabx)
        dimtabxn,dimtabyn,tabLung3d=wtebres(dirf,tabresLung,dimtabx,slicepitch,lung_name_gen,'lung')
        
        patch_list=pavgene(dirf,dimtabxn,dimtabx,tabScan3d,tabLung3d,dimtabyn,jpegpath3d)
        model=modelCompilation('front')
        proba=ILDCNNpredict(patch_list,model)
        pickle.dump(proba, open( os.path.join(dirf,"proba_front"), "wb" ))
        pickle.dump(patch_list, open( os.path.join(dirf,"patch_list_front"), "wb" ))

        visua(dirf,proba,patch_list,dimtabxn,dimtabx,dimtabyn,predictout3d,sroid,transbmp)
#        
        proba_cross=pickle.load(open( os.path.join(dirf,"proba_cross"), "rb" ))
        patch_list_cross=pickle.load(open( os.path.join(dirf,"patch_list_cross"), "rb" ))
        proba_front=pickle.load(open( os.path.join(dirf,"proba_front"), "rb" ))
        patch_list_front=pickle.load(open( os.path.join(dirf,"patch_list_front"), "rb" ))

        genethreef(dirf,patch_list_cross,proba_cross,slicepitch,dimtabx,dimtabx,dimpavx,slnt,'cross')
        genethreef(dirf,patch_list_front,proba_front,avgPixelSpacing,dimtabxn,dimtabyn,dimpavx,dimtabx,'front')
        proba_cross_gene,patch_list_cross_gene=genecross(proba_front,patch_list_front,slnt,slicepitch,dimtabxn,dimtabyn)
        visua(dirf,proba_cross_gene,patch_list_cross_gene,dimtabx,dimtabx,slnt,predictout3dr,sroi,scan_bmp)
        proba_merge,patch_list_merge=mergeproba(proba_cross,patch_list_cross,proba_cross_gene,patch_list_cross_gene,slnt,dimtabx,dimtabx)
#        print proba_merge
        visua(dirf,proba_merge,patch_list_merge,dimtabx,dimtabx,slnt,predictoutmerge,sroi,scan_bmp)
        genethreef(dirf,patch_list_merge,proba_merge,slicepitch,dimtabx,dimtabx,dimpavx,slnt,'merge')

    errorfile.write('completed :'+f)
    errorfile.close()  
errorfile.close()      
print "predict time:",round(mytime()-t0,3),"s"
  
#bglist=listcl()
#ILDCNNpredict(bglist)