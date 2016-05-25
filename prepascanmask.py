# coding: utf-8
#Sylvain Kritter 24 mai 2016
"""From scan image, generates patches bmp files according to patient database\
 and lung mask.\
 A patch is considered valid if the recovering area is above a threshold """
import g
import os
import numpy as np

import scipy.misc

from PIL import Image


errorfile = open(g.path_patient+'/prepscanerrorfile.txt', 'w')
px=g.dimpavx
py=g.dimpavy
dx=g.dimtabx
dy=g.dimtaby
mini=dx-px
minj=dy-py
pxy=float(px*py)



def pavgene (img,tabim,tablung,slicenumber):
    """ generate patches from scan"""
 
    tabf = np.copy(tabim)
# 
    i=0
    while i <= mini:
        j=0
#        j=maxj
        while j<=minj:
#            print(i,j)
            area=0.0
            x=0
            while x < px:
                y=0
                while y < py:
                   if tablung[y+j][x+i] >0:
                       area = area+1
                   y+=1
                x+=1
   
            #check if area above threshold
            if area/pxy>g.thrpatch:
     
                crorig = img.crop((i, j, i+px, j+py))
                imagemax=crorig.getbbox()
#               detect black patch
#                print (imagemax)
                if imagemax!=None:
                    crorig.save(patchpathf+'/p_'+slicenumber+'_'+str(i)+'_'+\
                           str(j)+'.'+g.typei)
                           #we draw the rectange
                    x=0
                    while x < px:
                        y=0
                        while y < py:
                            tabf[y+j][x+i]=[255,0,0]
                            if x == 0 or x == px-1 :
                                y+=1
                            else:
                                y+=py-1
                        x+=1
            
            j+=py    
        i+=px
#    im = plt.matshow(tabf)
#    plt.colorbar(im,label='with pavage')
    scipy.misc.imsave(jpegpathf+'/'+'s_'+slicenumber+'.jpg', tabf)



for f in g.patient_list:
    #f = 35
        print('generate patches on: ',f)
        namedirtopcf=os.path.join(g.path_patient,f)
#        print namedirtopcf
        namemask1=os.path.join(namedirtopcf,g.lungmask)
        namemask=os.path.join(namemask1,g.lungmaskbmp)
#        print namemask
        bmpdir = os.path.join(namedirtopcf,g.scanbmp)
        patchpathf=os.path.join(namedirtopcf,g.patchpath)
        jpegpathf=os.path.join(namedirtopcf,g.jpegpath)
        g.remove_folder(patchpathf)
        os.mkdir(patchpathf)
        g.remove_folder(jpegpathf)
        os.mkdir(jpegpathf)
        listbmp= os.listdir(bmpdir)
#        print(listbmp)
        if os.path.exists(namemask):
                listlungbmp= os.listdir(namemask)
              
        else:
            tflung=False
            listlungbmp=[]
#        print(listlungbmp)
        for img in listbmp:
#             print img
             endnumslice=img.find('.bmp')
             posend=endnumslice
             while img.find('-',posend)==-1:
                     posend-=1
             debnumslice=posend+1
             slicenumber=(img[debnumslice:endnumslice])
#             print('sln:',slicenumber,'img:', img,debnumslice,endnumslice           
             slns='_'+str(int(slicenumber))+'.'+g.typei
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
                    tablung = np.ones((dx, dy), dtype='i')
                     
             bmpfile = os.path.join(bmpdir,img)
             im = Image.open(bmpfile)
             imc= im.convert('RGB')
             tabim = np.array(imc)         
             pavgene (im,tabim,tablung,slicenumber)
            
errorfile.close()
print('completed')
   
