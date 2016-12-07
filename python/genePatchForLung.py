# coding: utf-8
#Sylvain Kritter 28 septembre 2016
"""Top file to generate patches from DICOM database for lung cv2 equalization
only patch with more than thblack% pixels not zero"""
import os
import numpy as np
import shutil
import scipy as sp
import scipy.misc
import dicom
#import PIL
#from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt    
#general parameters and file, directory names
#######################################################
#customisation part for datataprep
#patch size in pixels 32 * 32
dimpavx = 15
dimpavy = 15
imageDepth =255
#contrast
normiInternal=False
globalHist=True #use histogram equalization on full image
#threshold for patch acceptance
thr = 0.9
#threshold for non black pixel to be taken in patch output
thblack=0.6
#global directory for scan file
namedirHUG = 'CHU'
avgPixelSpacing=0.734   # average pixel spacing
subHUG='UIPTfull_s'
#subHUG='database'
#define the name of directory for patches
extendir='15_set4_g'
topreserv=False #special loca name to recognise and not subsample

#define the name of directory for patches
patchesdirname = 'patches_'+extendir
#define the name of directory for normalised patches
patchesNormdirname = 'patches_norm_'+extendir
#define the name for jpeg files
imagedirname='patches_jpeg_'+extendir
#define name for image directory in patient directory 
bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='bmp'
#localisation
if topreserv:
    loca=namedirHUG.lower()+'pr'
else:
    loca=namedirHUG.lower()
    

#directory created for lung only
dblung='dblung'
#directory created for area without lung
dbNolung='dbNolung'

#full path names
cwd=os.getcwd()
print cwd
(cwdtop,tail)=os.path.split(cwd)
print cwdtop
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
#print path_HUG
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
#print 'namediretc :',namedirtopc
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!') 

#end dataprep part
#########################################################
# general
#image  patch format
typei='bmp' #can be jpg

#end customisation part for datataprep
#######################################################
#color of labels
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)



classif ={
'nolung':0,
'lung':1,
  }

classifc ={
'nolung':yellow,
'lung':red,

 }
#print namedirtopc
#create patch and jpeg directory
patchpath=os.path.join(path_HUG,patchesdirname)

#create patch and jpeg directory
patchNormpath=os.path.join(path_HUG,patchesNormdirname)

#print patchpath
#define the name for jpeg files
jpegpath=os.path.join(path_HUG,imagedirname)
#print jpegpath


#patchpath = final/patches
if not os.path.isdir(patchpath):    
    os.mkdir(patchpath) 
for cl in classif:
        lp=os.path.join(patchpath,cl)
        if not os.path.isdir(lp):
            os.mkdir(lp)   
        lpl=os.path.join(lp,loca)
        if not os.path.isdir(lpl):
                os.mkdir(lpl)
                
if not os.path.isdir(patchNormpath):    
    os.mkdir(patchNormpath) 
for cl in classif:
        lp=os.path.join(patchNormpath,cl)
        if not os.path.isdir(lp):
            os.mkdir(lp)   
        lpl=os.path.join(lp,loca)
        if not os.path.isdir(lpl):
                os.mkdir(lpl)

        
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)   


#end log files
def fidclass(numero):
    """return class from number"""
    found=False
    for cle, valeur in classif.items():
        
        if valeur == numero:
            found=True
            return cle
      
    if not found:
        return 'unknown'

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)




def genebmp_lung(dirName):
    """generate images with lung and not lung"""
    print ('calculate bmp in :',f)
    #directory for images with and without lung
#    print dirName
    global dimtabx,dimtaby
#    dblung_dir = os.path.join(dirName, dblung)
#    remove_folder(dblung_dir)    
#    os.mkdir(dblung_dir)
#    dbNolung_dir = os.path.join(dirName, dbNolung)
#    remove_folder(dbNolung_dir)    
#    os.mkdir(dbNolung_dir)


    dirnamebmp = os.path.join(dirName, bmpname)
    remove_folder(dirnamebmp)    
    os.mkdir(dirnamebmp)
   

    #list dcm files
   
    listscanfile=[name for name in os.listdir(dirName) if \
                  name.lower().find('.dcm',0)>0]
    lung_mask_dir = os.path.join(dirName, lungmask)

    lung_dir_bmp = os.path.join(lung_mask_dir, lungmaskbmp)
    if not os.path.exists(lung_dir_bmp):
        remove_folder(lung_dir_bmp)    
        os.mkdir(lung_dir_bmp)

    listlungfile=[name for name in os.listdir(lung_mask_dir) if \
                  name.lower().find('.dcm',0)>0]
    
#    lunglist = os.listdir(lung_dir)
#    print '1'
    for filename in listscanfile:
#        print(filename)
        path_scan=os.path.join(dirName,filename)   
        RefDs = dicom.read_file(path_scan)       
        dsr= RefDs.pixel_array
    #scale the dicom pixel range to be in 0-255
        dsr= dsr-dsr.min()
        c=float(imageDepth)/dsr.max()
        dsr=dsr*c
        if imageDepth <256:
           dsr=dsr.astype('uint8')
        else:
               dsr=dsr.astype('uint16')
    #resize the dicom to have always the same pixel/mm
        fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing  
#        print 'scan',fxs
        dsrresize1= sp.misc.imresize(dsr,fxs,interp='bicubic',mode=None)
        if globalHist:
            dsrresize = cv2.equalizeHist(dsrresize1) 
        else:
            dsrresize=dsrresize1
    #calculate the  new dimension of scan image
        dimtabx=int(dsrresize.shape[0])
        dimtaby=int(dsrresize.shape[1])
#            print 'scan instance number',RefDs.InstanceNumber
#            scanNumber=rsliceNum(ll,'-','.dcm')
        scanNumber=int(RefDs.InstanceNumber)
        endnumslice=filename.find('.dcm')
        imgcore=filename[0:endnumslice]+'-'+str(scanNumber)+'.'+typei
        bmpfile=os.path.join(dirnamebmp,imgcore)
        scipy.misc.imsave(bmpfile,dsrresize)
        
    for filenamelung in listlungfile:
#        print('lung',filename)
        path_lung=os.path.join(lung_mask_dir,filenamelung)   
        RefDslung = dicom.read_file(path_lung)
       
        dsr= RefDslung.pixel_array
    #scale the dicom pixel range to be in 0-255
        dsr= dsr-dsr.min()
        scanNumber=int(RefDslung.InstanceNumber)
        endnumslice=filenamelung.find('.dcm')
        imgcore=filenamelung[0:endnumslice]+'_'+str(scanNumber)+'.'+typei
#        print 'lung',filename,imgcore
        fxslung=float(RefDslung.PixelSpacing[0])/avgPixelSpacing 
#        print filename,fxslung,fxs
        bmpfile=os.path.join(lung_dir_bmp,imgcore)
        if dsr.max() >0:
            c=float(imageDepth)/dsr.max()
            dsr=dsr*c
            if imageDepth <256:
               dsr=dsr.astype('uint8')
            else:
                   dsr=dsr.astype('uint16')
            dsrresize= sp.misc.imresize(dsr,fxslung,interp='bicubic',mode=None)
        else:
           dsrresize = np.zeros((dimtabx, dimtaby,3), dtype='i')
#        print bmpfile
        scipy.misc.imsave(bmpfile,dsrresize) 

#                 print 'end bmp'
def normi(img):
     tabi = np.array(img)
#     print(tabi.min(), tabi.max())
     tabi1=tabi-tabi.min()
#     print(tabi1.min(), tabi1.max())
     tabi2=tabi1*(255/float(tabi1.max()-tabi1.min()))
#     print(tabi2.min(), tabi2.max())   
     return tabi2
    
def pavlung(dirName,cla):
#    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    print 'pav lung in:', dirName, 'class :', cla
    """ generate patches from lung"""
    lung_dir1 = os.path.join(dirName, lungmask)
    lung_dir = os.path.join(lung_dir1, lungmaskbmp)
    listlung = os.listdir(lung_dir)
    
#    dblung_dir = os.path.join(dirName, dblung)
    dirdbname=os.path.join(dirName,bmpname)
    ldb=os.listdir(dirdbname)
#    print ldb
    label=cla

    pxy=dimpavx*dimpavy*255
    for  n in listlung:    
            print n
            tabp = np.zeros((dimtabx, dimtaby,3), dtype='i')

          
            endnumslice=n.find('.bmp')           
            posend=endnumslice
#            print n
            while n.find('_',posend)==-1:
                posend-=1
            debnumslice=posend+1
            slicenumberl=int((n[debnumslice:endnumslice])) 
#            print slicenumberl
            for ld in ldb:
                endnumslice=ld.find('.bmp')           
                posend=endnumslice
                while ld.find('-',posend)==-1:
                  posend-=1
                debnumslice=posend+1
                slicenumbers=int((ld[debnumslice:endnumslice])) 
#                print slicenumbers, slicenumberl
                if slicenumbers==slicenumberl:
#                    print slicenumbers
                    filescan=os.path.join(dirdbname,ld)
#                    print filescan
                    break
#            print filescan
            lungfile = os.path.join(lung_dir, n)
            imglung=cv2.imread(lungfile,1)
#            print filescan
            
            img1 = cv2.imread(filescan,1)
            
            img2 = cv2.medianBlur(imglung,9)
#            cv2.imshow('image',img2)
#            cv2.waitKey(0)
            imgray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(imgray,0,255,0)
            if label=='nolung':
                  thresh = cv2.bitwise_not(thresh)            
            
#            cv2.imshow('image',thresh)
#            cv2.waitKey(0)
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

#                        print dimpavx*dimpavy*thblack, dimpavx*dimpavy
                        if imagemax> dimpavx*dimpavy*thblack and min_val!=max_val:
#                            print imagemax,dimpavx*dimpavy/2
                            nbp+=1
                            nampa=f+'_'+str(slicenumbers)+'_'+str(nbp)+'.'+typei 
#                            print nampa
                            fw=os.path.join(patchpath,label)
                            fxloca=os.path.join(fw,loca)
                            fw1=os.path.join(fxloca,nampa)
#                            print fw1
#                            ooo
                            scipy.misc.imsave(fw1, imgray)
                            if normiInternal:
                                        normpatch = normi(imgray) 
                            else:
                                        normpatch = cv2.equalizeHist(imgray) 
#                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)
#                            print (min_val, max_val,min_loc, max_loc)                                                                                   
#                            normpatch = cv2.equalizeHist(imgray)
#normalize patches and put in patches_norm
                            fw=os.path.join(patchNormpath,label)
                            fxloca=os.path.join(fw,loca)
                            fw1=os.path.join(fxloca,nampa)
                            cv2.imwrite(fw1,normpatch)
                                                        
                        #                print('pavage',i,j)  
                            i=0
                            #we draw the rectange
                            col=classifc[label]
                            while i < dimpavx:
                                j=0
                                while j < dimpavy:
                                    if y+j<512 and x+i<512:
                                        tabp[y+j][x+i]=col
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

            mfl=open(jpegpath+'/'+f+'_'+str(slicenumbers)+'.txt',"w")
            mfl.write('#number of patches: '+str(nbp)+'\n')
            mfl.close()
            scipy.misc.imsave(jpegpath+'/'+f+'_'+label+'_'+str(slicenumbers)+'.jpg', tabp)



listdirc= os.listdir(namedirtopc)
npat=0
for f in listdirc:
    #f = 35
    namedirtopcf=namedirtopc+'/'+f
    if os.path.isdir(namedirtopcf):
        print('work on:',f)
    
    
       
        #namedirtopcf = final/ILD_DB_txtROIs/35
        genebmp_lung(namedirtopcf)
        for cla in classif:
            pavlung(namedirtopcf,cla)


print('completed')