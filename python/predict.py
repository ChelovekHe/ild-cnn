# coding: utf-8
#Sylvain Kritter 24 mai 2016
"""general parameters and file, directory names"""

import os
import dicom
import scipy
from scipy import misc
import shutil
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cPickle as pickle
import ild_helpers as H
import cnn_model as CNN4

#########################################################
# for predict
#to enhance contrast on patch put True
contrast=False
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

#px=g.dimpavx
#py=g.dimpavy
#dx=g.dimtabx
#dy=g.dimtaby
mini=dimtabx-dimpavx
minj=dimtaby-dimpavy

pxy=float(dimpavx*dimpavy)
#threshold for patch acceptance
thrpatch = 0.8
#threshold for probability prediction
thrproba = 0.9

#end general part
#font file imported in top directory
# font = ImageFont.truetype( 'arial.ttf', 20)
font = ImageFont.truetype( '../fonts/arial.ttf', 20)
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

classif ={
'consolidation':0,
'fibrosis':1,
'ground_glass':2,
'healthy':3,
'micronodules':4,
'reticulation':5}

classifc ={
'consolidation':red,
'fibrosis':blue,
'ground_glass':yellow,
'healthy':green,
'micronodules':cyan,
'reticulation':purple}

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
        # 
             i=0
             while i <= mini:
                 j=0
        #        j=maxj
                 while j<=minj:
        #            print(i,j)
                     area=0.0
                     x=0
                     while x < dimpavx:
                        y=0
                        while y < dimpavy:
                           if tablung[y+j][x+i] >0:
                               area +=1
                           y+=1
                        x+=1           
                    #check if area above threshold
                     if area/pxy>thrpatch:
             
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

    deltay=25*(classif[text]%3)
    deltax=175*(classif[text]//3)
    #print text, col
    draw.text((x+deltax, y+deltay),text+' '+pro,col,font=font)

    imgn.save(fig)

def tagviews(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font)
    imgn.save(fig)

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in interv(0,lenp-1):
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
#    dd = open(preclasspick,'rb')
#    my_depickler = pickle.Unpickler(dd)
#    preclass = my_depickler.load()
#   
#    dd.close()
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


        
def scanx(tab):
    tabh= np.zeros((dimtabx, dimtaby), dtype='i')
    for x in interv(1,dimtabx-1):
        for y in interv(1,dimtaby-1):
            if tab[x][y-1]==0 and tab[x][y]>0:
                tabh[x][y]=tab[x][y]
            elif tab[x][y-1]==0 and tab[x][y]==0:
                tabh[x][y]=0
            elif tab[x][y-1]>0 and tab[x][y]==0:
              tabh[x][y-1]=tab[x][y-1]
            elif tab[x][y-1]>0 and tab[x][y]>0:
                if tab[x][y-1] == tab[x][y]:
                    tabh[x][y]=0
                else:
                    tabh[x][y]=tab[x][y] 
                    tabh[x][y-1]=tab[x][y-1]

    return tabh
    
def scany(tab):
    tabh= np.zeros((dimtabx, dimtaby), dtype='i')
    for y in interv(1,dimtaby-1):
        for x in interv(1,dimtabx-1):
            if tab[x-1][y]==0 and tab[x][y]>0:
                tabh[x][y]=tab[x][y]
            elif tab[x-1][y]==0 and tab[x][y]==0:
                tabh[x][y]=0
            elif tab[x-1][y]>0 and tab[x][y]==0:
              tabh[x-1][y]=tab[x-1][y]
            elif tab[x-1][y]>0 and tab[x][y]>0:
                if tab[x-1][y] == tab[x][y]:
                    tabh[x][y]=0
                else:
                    tabh[x][y]=tab[x][y]
                    tabh[x-1][y]=tab[x-1][y]
    return tabh


    return tabh         
def merg(tabs,tabp):
    tabh= np.zeros((dimtabx, dimtaby), dtype='i')
    for y in interv(0,dimtaby-1):
        for x in interv(0,dimtabx-1):
            if tabp[x][y]>0:
                tabh[x][y]=tabp[x][y]
            else:
                tabh[x][y]=tabs[x][y]
    return tabh 

def contour(tab):
     tabx=scanx(tab)
     taby=scany(tab)
     tabi=merg(tabx,taby)
     return tabi


def mergcolor(tabs,tabp):
    tabh= np.zeros((dimtabx, dimtaby,3), dtype='i')
    
    for y in interv(0,dimtaby-1):
        for x in interv(0,dimtabx-1):
            
            if tabp[x][y]>0:
                prec= tabp[x][y]-100
                classlabel=fidclass(prec)
                classcolor=classifc[classlabel]
                tabh[x][y]=classcolor
            else:
                tabh[x][y]=tabs[x][y]
    return tabh 
    
def  visua(dirpatientdb):
    print('visualise work on: ',f)
    #directory name with predict out dabasase, will be created in current directory
    predictout_dir = os.path.join(dirpatientdb, predictout)
    remove_folder(predictout_dir)
    os.mkdir(predictout_dir)   
    (preprob,listnamepatch)=loadpkl(dirpatientdb)

    dirpatientfdb=os.path.join(dirpatientdb,scanbmp)
    dirpatientfsdb=os.path.join(dirpatientdb,sroi)
    listbmpscan=os.listdir(dirpatientfdb)
    listlabelf={}
#    setname=f
    
    for img in listbmpscan:
        listlabel={}
        listlabelaverage={}
        if os.path.exists(dirpatientfsdb):
#        imgc=os.path.join(dirpatientfdb,img)
            imgc=os.path.join(dirpatientfsdb,img)
        else:
            imgc=os.path.join(dirpatientfdb,img)

#        print img  
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

    #initialise index in list of results
        ill = 0
      
        foundp=False
        tabsi = np.zeros((dimtabx, dimtaby), dtype='i')
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

        #we found label from prediction
#            prec=int(preclass[ill])
         
        #we found max proba from prediction
            proba=preprob[ill]
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
            classlabel=fidclass(prec)
#            print(mproba)
            #print(setname, slicename,xpat,ypat,classlabel,classcolor,mproba)
              
            if mproba >thrproba and slicenumber == slicename:
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
#                        listlabelaverage[classlabel]=mproba
                 
#                        listlabel.append((classlabel,mproba))
                    x=0
                    while x < dimpavx:
                        y=0
                        while y < dimpavy:
                            tabsi[y+ypat][x+xpat]=prec+100
                            y+=1    
                        x+=1
            ill+=1
  
        tablscanc =mergcolor(tablscan,contour(tabsi))
        imgcorefull=imgcore+'.bmp'
        imgname=os.path.join(predictout_dir,imgcorefull)
        scipy.misc.imsave(imgname, tablscanc)
        textw='n: '+f+' scan: '+str(slicenumber)
        tagviews(imgname,textw,0,20)
        if foundp:
            tagviews(imgname,'average probability',0,0)           
            for ll in listlabel:
                tagview(imgname,ll,str(listlabelaverage[ll]),175,00)
        else:   
            tagviews(imgname,'no recognised label',0,0)
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
errorfile.close() 
