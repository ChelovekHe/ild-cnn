# coding: utf-8
#Sylvain Kritter 24 mai 2016
"""From predicted patches from scan image, visualize results as image overlay """
import g
import os
import numpy as np


import scipy.misc
import cPickle as pickle
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw

#font file imported in top directory
font = ImageFont.truetype( 'arial.ttf', 20)


print('hello, world')

#error file
errorfile = open(g.path_patient+'/visua_errorfile.txt', 'w')
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
    for i in g.interv(0,lenp-1):
        if proba[i]>m:
            m=proba[i]
    return m

    

def loadpkl(do):
    """crate image directory and load pkl files"""
    dop =os.path.join(do,g.picklefile)
    #pickle with predicted classes
    preclasspick= os.path.join(dop,g.predicted_classes)
    #pickle with predicted probabilities
    preprobpick= os.path.join(dop,g.predicted_proba)
     #pickle with xfileref
    prexfilepick= os.path.join(dop,g.Xrefpkl)
    """generate input tables from pickles"""
    dd = open(preclasspick,'rb')
    my_depickler = pickle.Unpickler(dd)
    preclass = my_depickler.load()
#   
    dd.close()
    dd = open(preprobpick,'rb')
    my_depickler = pickle.Unpickler(dd)
    preprob = my_depickler.load()
    dd.close()  
    dd = open(prexfilepick,'rb')
    my_depickler = pickle.Unpickler(dd)
    prexfile = my_depickler.load()
    dd.close()  
    return (preclass,preprob,prexfile)
    

for f in g.patient_list:
    print('visualise work on: ',f)
    listlabelf={}
    listprobaf={}
    nlabel=0
    #directory name with patient databases, 
    dirpatientdb = os.path.join(g.path_patient,f)

    #directory name with patch db databases, should be in current directory
#    dirpatientpatchdb = os.path.join(cwd,'predict')

    #directory name with predict out dabasase, will be created in current directory
    predictout_dir = os.path.join(dirpatientdb, g.predictout)
    g.remove_folder(predictout_dir)
    os.mkdir(predictout_dir)   

    (preclass,preprob,listnamepatch)=loadpkl(dirpatientdb)


    dirpatientfdb=os.path.join(dirpatientdb,g.scanbmp)

    listbmpscan=os.listdir(dirpatientfdb)

#    setname=f
    
    for img in listbmpscan:
        listlabel={}
        listlabelaverage={}
        imgc=os.path.join(dirpatientfdb,img)
#        print img  
        endnumslice=img.find('.'+g.typei)
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
#        for i in g.interv(0,25):
#           for j in g.interv (0,511):
#              tablscan[i][j]=0
    #initialise index in list of results
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

        #we found label from prediction
            prec=int(preclass[ill])
            classlabel=fidclass(prec)
            classcolor=classifc[classlabel]
        #we found max proba from prediction
            proba=preprob[ill]
            mproba=round(maxproba(proba),2)
#            print(mproba)
            #print(setname, slicename,xpat,ypat,classlabel,classcolor,mproba)
            ill+=1  
            if mproba >g.thrproba and slicenumber == slicename:
#                    print(setname, slicename,xpat,ypat,classlabel,classcolor,mproba)
                    
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
#                        cur=listlabelaverage[classlabel]
##                               print (numl,cur)
#                        averageproba= round((cur*numl+mproba)/(numl+1),2)
#                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabelf[classlabel]=1
#                        listlabelaverage[classlabel]=mproba
                 
#                        listlabel.append((classlabel,mproba))
                    x=0
                    while x < g.dimpavx:
                        y=0
                        while y < g.dimpavy:
                            tablscan[y+ypat][x+xpat]=classcolor
                            if x == 0 or x == g.dimpavx-1 :
                                y+=1
                            else:
                                y+=g.dimpavy-1
                        x+=1
                        #tablscans=tablscan[:,:,1]
                        #im = plt.matshow(tablscans)
                        #plt.colorbar(im,label='scan mask')
         #plt.show()

#        nlt=0     
#        for classlabel in listlabel:  
#           print('patient: ',f,'scan:',slicenumber,'label:',classlabel,': ',listlabel[classlabel])

        
        imgcorefull=imgcore+'.jpg'
        imgname=os.path.join(predictout_dir,imgcorefull)
        scipy.misc.imsave(imgname, tablscan)
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
errorfile.close()                            
print('completed')
            
