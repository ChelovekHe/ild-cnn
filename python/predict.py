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

#########################################################
# for predict
#to enhance contrast on patch put True
contrast=True
#threshold for patch acceptance
thrpatch = 0.8
#threshold for probability prediction
thrproba = 0.9
#global directory for predict file
namedirtop = 'predict_essai'

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
picklefile='pickle'

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)

path_patient = os.path.join(cwdtop,namedirtop)
#print PathDicom
patient_list= os.walk(path_patient).next()[1]
#print patient_list
# list label not to visualize
excluvisu=['back_ground','healthy']
#dataset supposed to be healthy
datahealthy=['138']
#end predict part
#########################################################
# general
#image  patch format
typei='bmp' #can be jpg
#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx =32
dimpavy = 32

mini=dimtabx-dimpavx
minj=dimtaby-dimpavy

pxy=float(dimpavx*dimpavy)

#end general part
#font file imported in top directory
font20 = ImageFont.truetype( 'arial.ttf', 20)
font10 = ImageFont.truetype( 'arial.ttf', 10)
#########################################################
errorfile = open(path_patient+'/predictlog.txt', 'w') 

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

   
def genebmp(dirName):
    """generate patches from dicom files"""
    print ('generate  bmp files from dicom files in :',f)
    #directory for patches
  
    bmp_dir = os.path.join(dirName, scanbmp)
#    print bmp_dir
    remove_folder(bmp_dir)    
    os.mkdir(bmp_dir)    
    #list dcm files
    fileList = os.listdir(dirName)
    for filename in fileList:
#        print(filename)
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            FilesDCM =(os.path.join(dirName,filename))  
#           
            ds = dicom.read_file(FilesDCM)
            endnumslice=filename.find('.dcm')
            imgcore=filename[0:endnumslice]+'.'+typei
#            imgcore=filename[0:endnumslice]+'.dcm'

#            print imgcore
            bmpfile=os.path.join(bmp_dir,imgcore)
            scipy.misc.imsave(bmpfile, ds.pixel_array)
            if np.shape(ds.pixel_array)[0] !=dimtabx:
                orig= Image.open(bmpfile,'r')             
                ncr=orig.resize((dimtabx,dimtaby),PIL.Image.ANTIALIAS)
                del orig
                ncr.save(bmpfile)
#            print np.shape(ds.pixel_array)
                
#            ds.save_as(bmpfile)

        #chek if lung mask present       
        if lungmask == filename:
         
             lung_dir = os.path.join(dirName, lungmask)
             lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
             lunglist = os.listdir(lung_dir)
             remove_folder(lung_bmp_dir)
#             if lungmaskbmp not in lunglist:
             os.mkdir(lung_bmp_dir)
#             print(lung_bmp_dir)
             for lungfile in lunglist:
#                print(lungfile)
                if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                    lungDCM =os.path.join(lung_dir,lungfile)  
                    dslung = dicom.read_file(lungDCM)
                    endnumslice=lungfile.find('.dcm')
                    lungcore=lungfile[0:endnumslice]+'.'+typei
                    lungcoref=os.path.join(lung_bmp_dir,lungcore)
                    scipy.misc.imsave(lungcoref, dslung.pixel_array)
                    if np.shape(dslung.pixel_array)[0] !=dimtabx:
                        orig= Image.open(lungcoref,'r')
                        ncr=orig.resize((dimtabx,dimtaby),PIL.Image.ANTIALIAS)
                        del orig
                        ncr.save(lungcoref)
                        

def normi(img):
     """ normalise patches 0 255"""
     tabi = np.array(img)
#     print(tabi.min(), tabi.max())
     tabi1=tabi-tabi.min()
#     print(tabi1.min(), tabi1.max())
     tabi2=tabi1*(255/float(tabi1.max()-tabi1.min()))
#     print(tabi2.min(), tabi2.max())
     return tabi2

def pavgene (namedirtopcf):
        """ generate patches from scan"""
        print('generate patches on: ',f)
#        print namedirtopcf
        namemask1=os.path.join(namedirtopcf,lungmask)
        namemask=os.path.join(namemask1,lungmaskbmp)
#        print namemask
        bmpdir = os.path.join(namedirtopcf,scanbmp)
#        print bmpdir
        patchpathf=os.path.join(namedirtopcf,patchpath)
        jpegpathf=os.path.join(namedirtopcf,jpegpath)
        remove_folder(patchpathf)
        os.mkdir(patchpathf)
        remove_folder(jpegpathf)
        os.mkdir(jpegpathf)
        listbmp= os.listdir(bmpdir)
#        print(listbmp)
        if os.path.exists(namemask):
                listlungbmp= os.listdir(namemask)            
        else:
            tflung=False
            listlungbmp=[]
        for img in listbmp:
#             print img
             endnumslice=img.find('.bmp')
             posend=endnumslice
             while img.find('-',posend)==-1:
                     posend-=1
             debnumslice=posend+1
             slicenumber=(img[debnumslice:endnumslice])
#             print('sln:',slicenumber,'img:', img,debnumslice,endnumslice           
             slns='_'+str(int(slicenumber))+'.'+typei
#             print(slns)
             for llung in listlungbmp:
                tflung=False
#                print(llung)
#                print(listlungbmp)

                if llung.find(slns) >0:
                    tflung=True
                    lungfile = os.path.join(namemask,llung)
#                    print(lungfile)
                    imlung = Image.open(lungfile)
                    tablung = np.array(imlung)
                    np.putmask(tablung,tablung>0,1)

                    break
             if not tflung:
                    errorfile.write('lung mask not found '+slns+' in: '+f) 
                    print('lung mask not found ',slns,' in: ',f)
                    tablung = np.ones((dimtabx, dimtaby), dtype='i')
                     
             bmpfile = os.path.join(bmpdir,img)
             im = Image.open(bmpfile)
             imc= im.convert('RGB')
             tabf = np.array(imc)         
#             pavgene (im,tabim,tablung,slicenumber)
           
             nz= np.count_nonzero(tablung)
             if nz>0:
            
                atabf = np.nonzero(tablung)
                #tab[y][x]  convention
                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()
             else:
                xmin=0
                xmax=0
                ymin=0
                ymax=0
                
#             atabf = np.nonzero(tablung)
#        #tab[y][x]  convention
#             xmin=atabf[1].min()
#             xmax=atabf[1].max()
#             ymin=atabf[0].min()
#             ymax=atabf[0].max()
             i=xmin
             while i <= xmax:
                 j=ymin
        #        j=maxj
                 while j<=ymax:
        #            print(i,j)
                     tabpatch=tablung[j:j+dimpavy,i:i+dimpavx]
                     area= tabpatch.sum()  
        
#                    check if area above threshold
                     targ=float(area)/pxy
                     if targ>thrpatch:
             
                        crorig = im.crop((i, j, i+dimpavx, j+dimpavy))
                        imagemax=crorig.getbbox()
        #               detect black patch
        #                print (imagemax)
                        if imagemax!=None:
                            namepatch=patchpathf+'/p_'+slicenumber+'_'+str(i)+'_'+str(j)+'.'+typei
                            if contrast:
                                    tabcont=normi(crorig)
                                    scipy.misc.imsave(namepatch, tabcont)
                            else:
                                crorig.save(namepatch)
                                   #we draw the rectange
                            x=0
                            while x < dimpavx:
                                y=0
                                while y < dimpavy:
                                    tabf[y+j][x+i]=[255,0,0]
                                    if x == 0 or x == dimpavx-1 :
                                        y+=1
                                    else:
                                        y+=dimpavy-1
                                x+=1                    
                     j+=dimpavy    
                 i+=dimpavx
        #    im = plt.matshow(tabf)
        #    plt.colorbar(im,label='with pavage')
             scipy.misc.imsave(jpegpathf+'/'+'s_'+slicenumber+'.bmp', tabf)
        
def dataprocessing(patient_dir_s):
    
    print ('predict data processing work on: ',f)

#    print(listcwd)
    patient_dir = os.path.join(patient_dir_s,patchpath)
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
        if fil.find(typei) > 0:  
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
    # this is already in greyscale 
#    X = dataset[:,:, :,1]
    file_reference = np.array(file_reference_list)
#   
    #dir to put pickle files
    predictout_f_dir = os.path.join( patient_dir_s,picklefile)
    #print predictout_f_dir
    remove_folder(predictout_f_dir)
    os.mkdir(predictout_f_dir)

    xfp=os.path.join(predictout_f_dir,Xprepkl)
    xfpr=os.path.join(predictout_f_dir,Xrefpkl)
    pickle.dump(X, open( xfp, "wb" ))
    pickle.dump(file_reference, open( xfpr, "wb" ))
# 
def ILDCNNpredict(patient_dir_s):     
        print ('predict work on: ',f)      
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
        model = H.load_model()
        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        
#    print patient_dir_s
        patient_dir_pkl= os.path.join(patient_dir_s, picklefile)
#        print patient_dir_pkl
        patient_pred_file =os.path.join( patient_dir_pkl,Xprepkl)
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
        xfproba=os.path.join( patient_dir_pkl,predicted_proba)
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


def tagview(fig,text,pro,x,y):
    """write text in image according to label and color"""
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=classifc[text]
    labnow=classif[text]
#    print (labnow, text)
    if labnow == 0:
        x=0
        y=0        
        deltax=0
        deltay=60
    else:        
        deltay=25*((labnow-1)%3)
        deltax=175*((labnow-1)//3)
    #print text, col
    draw.text((x+deltax, y+deltay),text+' '+pro,col,font=font20)

    imgn.save(fig) 
    
def tagviews(fig,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
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


def loadpkl(do):
    """crate image directory and load pkl files"""
    dop =os.path.join(do,picklefile)
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
#print (len(contours))
#cnt=contours[2]
        
        contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
        cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)

    return vis
#cv2.drawContours(im,contours,-1,(0,255,0),-1)
        

def  visua(dirpatientdb):
    print('visualise work on: ',f)
    
    #directory name with predict out dabasase, will be created in current directory
    predictout_dir = os.path.join(dirpatientdb, predictout)
    predictout_dir_th = os.path.join(predictout_dir,str(thrproba))
#    print predictout_dir_th
    if not os.path.exists(predictout_dir) :
         os.mkdir(predictout_dir)    
    remove_folder(predictout_dir_th)
    os.mkdir(predictout_dir_th)
    (preprob,listnamepatch)=loadpkl(dirpatientdb)

    dirpatientfdb=os.path.join(dirpatientdb,scanbmp)
    dirpatientfsdb=os.path.join(dirpatientdb,sroi)
    listbmpscan=os.listdir(dirpatientfdb)
    listlabelf={}
#    setname=f
#    tabsim1 = np.zeros((dimtabx, dimtaby), dtype='i')
    for img in listbmpscan:
#        print img
        #assume less than 100 patches per slice
        imgt = np.zeros((dimtabx,dimtaby,3), np.uint8)
        imn = np.zeros((dimtabx,dimtaby,3), np.uint8)
        listlabel={}
        listlabelaverage={}
        if os.path.exists(dirpatientfsdb):
#        imgc=os.path.join(dirpatientfdb,img)
            imgc=os.path.join(dirpatientfsdb,img)
        else:
            imgc=os.path.join(dirpatientfdb,img)

        
#                        orig= Image.open(lungcoref,'r')
#                        ncr=orig.resize((dimtabx,dimtaby),PIL.Image.ANTIALIAS)
#                        del orig
#                        ncr.save(lungcoref)        
#        print imgc  
        endnumslice=img.find('.'+typei)
        imgcore=img[0:endnumslice]
#        print imgcore
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


            if mproba >thrproba and slicenumber == slicename and\
            (f in datahealthy or (classlabel not in excluvisu)):
#                    print(setname, slicename,xpat,ypat,classlabel,classcolor,mproba)
#                    print(mproba,preclass[ill],preprob[ill])
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

        vis=drawContour(imgt,listlabel)
#        print tablscan.shape
        imn=cv2.add(vis,tablscan)
        imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
        imgcorefull=imgcore+'.bmp'
        imgname=os.path.join(predictout_dir_th,imgcorefull)
        cv2.imwrite(imgname,imn)
#        scipy.misc.imsave(imgname, tablscanc)
       
        if foundp:
            t0='average probability'
        else:
            t0='no recognised label'
        t1='n: '+f+' scan: '+str(slicenumber)        
        t2='CONFIDENTIAL - prototype - not for medical use'
        t3='threshold: '+str(thrproba)
        t4=time.asctime()
        tagviews(imgname,t0,0,0,t1,0,20,t2,20,485,t3,0,40,t4,0,50)
        if foundp:
#            tagviews(imgname,'average probability',0,0)           
            for ll in listlabel:
                tagview(imgname,ll,str(listlabelaverage[ll]),175,00)
        else:   
#            tagviews(imgname,'no recognised label',0,0)
            errorfile.write('no recognised label in: '+str(f)+' '+str (img)+'\n' )
#            print('no recognised label in: '+str(f)+' '+str (img) )
    errorfile.write('\n'+'number of labels in :'+str(f)+'\n' )
    for classlabel in listlabelf:  
          print('p: ',f,'label:',classlabel,': ',listlabelf[classlabel])
          string=str(classlabel)+': '+str(listlabelf[classlabel])+'\n' 
#          print string
          errorfile.write(string )

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
for f in patient_list:
    #f = 35
    print('work on:',f)
    namedirtopcf = os.path.join(path_patient,f)
#    print namedirtopcf
    if os.path.isdir(namedirtopcf):
        renomscan(namedirtopcf)
        genebmp(namedirtopcf)
        pavgene(namedirtopcf)
        dataprocessing(namedirtopcf)
        ILDCNNpredict(namedirtopcf)
        visua(namedirtopcf)
        print('completed on: ',f)
t = datetime.datetime.now()
if (t.minute)<9:
    tm='0'+str(t.minute)
else:
    tm=str(t.minute)
if (t.hour)<9:
    th='0'+str(t.hour)
else:
    th=str(t.hour)
today = str('date: '+dd(t.month)+'-'+dd(t.day)+'-'+str(t.year)+\
'_'+dd(t.hour)+':'+dd(t.minute)+':'+dd(t.second))
print today
errorfile.close() 
