# coding: utf-8
#Sylvain Kritter 26 mai 2016
"""Top file to generate patches from DICOM database"""
import os
import numpy as np
import shutil
import scipy.misc
import dicom
from PIL import Image
    
"""general parameters and file, directory names"""

#######################################################
#customisation part for datataprep
#global directory for scan file
namedirHUG = 'HUG'
#subdir for roi in text
subHUG='ILD_TXT'
#define the name of directory for patches
patchesdirname = 'patches'
#define the name for jpeg files
imagedirname='patches_jpeg'
#define name for image directory in patient directory 
bmpname='bmp'

#full path names
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!') 

#print namedirtopc
#create patch and jpeg directory
patchpath=os.path.join(cwdtop,patchesdirname)
#print patchpath
#define the name for jpeg files
jpegpath=os.path.join(cwdtop,imagedirname)
#print jpegpath

if not os.path.isdir(patchpath):
    os.mkdir(patchpath)   
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)   
#patchpath = final/patches

#end dataprep part
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
#threshold for patch acceptance
thrpatch = 0.8

#end general part
#########################################################
#log files
##error file
errorfile = open(namedirtopc+'genepatcherrortop.txt', 'w')
#filetowrite=os.path.join(namedirtopc,'lislabel.txt')
mf=open(namedirtopc+'lislabel.txt',"w")

#end customisation part for datataprep
#######################################################


#end log files
  
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
    print ('generate  bmp files from dicom files in :',dirName)
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
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
#            print imgcore
            bmpfile=os.path.join(bmp_dir,imgcore)
            scipy.misc.imsave(bmpfile, ds.pixel_array)

listlabel=[]
#print(namedirtopc)
listdirc= (os.listdir(namedirtopc))
#print(listcwd)
#print(listdirc)

def repts(tabc,dx,dy):
    """ we fill tab with summits of tabc"""
    tab = np.zeros((dx, dy), dtype='i')
    i=0
    tabcs0=tabc.shape[0]

    while i < tabcs0:
         x1 =min(511,tabc[i][0])
         y1 =min(511,tabc[i][1]) 
#         print(x1,y1)
         tab[y1][x1]=1
         if i<  tabcs0-1:
             i+=1
             x2 =min(511,tabc[i][0]) 
             y2 =min(511,tabc[i][1]) 
             tab[y2][x2]=1
         i+=1
    return tab

#tabz1 = np.zeros((10, 10), dtype='i')

def repti(tabc,dx,dy):
    """ draw line in between summits from tab"""
    tab = np.zeros((dx, dy), dtype='i')
    tabcs0=tabc.shape[0]
    i=0
    while i < tabcs0:
  
        x1 =min(511,tabc[i][0])
        y1 =min(511,tabc[i][1]) 
       # print(x1,y1)
        if i< tabcs0-1:
             i+=1
             x2 =min(511,tabc[i][0]) 
             y2 =min(511,tabc[i][1]) 
        else:
             i+=1
             x2 =min(511,tabc[0][0]) 
             y2 =min(511,tabc[0][1]) 
       # print('1',x1,y1,x2,y2)
        if x2-x1 != 0:
            
                 if x1>x2:
                     
                    x2,x1=x1,x2
                    y2,y1=y1,y2
                 xi=x1
                 c=float(y2-y1)/float(x2-x1)
#                 sh=False
                 while xi < x2:
#                     print xi, x2
                     yi = min(511,y1 + c * (xi-x1))
#                     if x1==117 and y1<230:
#                         sh=True
#                         print('c: ',c, 'xi:',xi,'x1:',x1,'y1:',y1,\
#                         ' x2:',x2,'y2:', y2, 'yi:',yi, 'round yi:',int(round(yi)))
                     yi = int(round(yi))
               
#                     print yi
                     #print('x1:',x1, 'y1:',y1, 'x2:',x2,'y2:',y2,'xi:',xi,'yi:',yi,'c:',c)             
                     tab[yi][int(xi)]=1
#                     xi+=1
                     if c!=0:
                         xi+=min(abs(c),abs(1/c))
#                         xi+=0.1
#                         print('delta: ',c,min(abs(c),abs(1/c)),xi)
                        
                     else:
                         xi+=1
#                     if sh:
#                             print(xi,x2)       
        else:
                if y1>y2:
                    y2,y1=y1,y2
                yi=y1
                while yi < y2:
         #           print('3',int(x1),int(yi))
                    tab[yi][x1]=1
                    yi+=1
    return tab


"""table
  y I 0
    I 1
    I 2 tab[y][x]
    I .
      0 1 2 3 .
    ---------
         x """

def reptf(tab,tabc,dx,dy):
    """ fill the full area of tab"""
    tabf = np.zeros((dx, dy), dtype='i')
    mintaby= min(510,tabc.take([1],axis=1).min())
    maxtaby= min(510,tabc.take([1],axis=1).max())
    mintabx= min(510,tabc.take([0],axis=1).min())
    maxtabx= min(510,tabc.take([0],axis=1).max())
#    print(mintabx,maxtabx,mintaby,maxtaby)
    x=mintabx

    while x <= maxtabx:
        y=mintaby 
        while y<=maxtaby:
            inst=False
            ycu=y
            xcu=x
            noyh=0
            noyl=0
            noxl=0
            noxr=0
            """ look right horizon"""
            while xcu <=maxtabx:
#                if tab[y][xcu] >0 and tab[y][xcu-1]==0:
                if tab[y][xcu] >0 and tab[y][xcu+1]==0:
                    noxr = noxr+1
                xcu+=1
            xcu=x
            """look left horiz"""
            while xcu >=mintabx:
             
                if tab[y][xcu] >0 and tab[y][xcu-1]==0:
                     noxl = noxl+1
                xcu-=1
#                if(x==9 and y==9):
#                    print(x,y,xcu,noxl)
          
            ycu=y
            """look high vertic"""
            while ycu <=maxtaby:
                if tab[ycu][x] >0 and tab[ycu+1][x]==0:
                    noyh = noyh+1
                ycu+=1
            ycu=y
            """look low vertic"""
            while ycu >=mintaby:
                if tab[ycu][x] >0 and tab[ycu-1][x]==0:
                    noyl = noyl+1
                ycu-=1
           
            if noyl ==0 or noyh ==0 or noxl==0 or noxr==0:
               #     a=1
               inst=False 
            else:
#                inst = True
                if (noyl %2 != 0 or noyh%2 != 0 or noxl%2 != 0 or
                noxr%2 != 0):
                    inst=True
            if inst :
                if (tab[y][x]==0):
                    tabf[y][x]=3
            y+=1
        x+=1
    x=1
    return tabf
    
def reptfc(tab,dx,dy):
    """ correct  the full area of tab from artefacts"""
    tabf = np.copy(tab)
    x=1
    while x < dx-1:
        y=1
        while y<dy-1 :
            if (( tabf[y+1][x]==0 or tabf[y][x+1]==0) and \
            tabf[y][x]==3):
                tabf[y][x]=0
               
            y+=1
        x+=1
    y=1
    while y < dy-1:
        x=1
        while x < dx-1 :
            if ((tabf[y][x+1]==0 or tabf[y+1][x]==0) and \
            tabf[y][x]==3):
                tabf[y][x]=0
                
            x+=1
        y+=1
    x=1
    while x < dx-1:
        y=1
        while y<dy-1 :
            if  tabf[y][x]>0 :
                tabf[y][x]=1
                
            y+=1
        x+=1
    return tabf


#tabz2=tabz1+reptf(tabz1,mon_tableauc)  
def reptfull(tabc,dx,dy):
    """ top function to generate ROI table filled from ROI text file"""
    tabz=repts(tabc,dx,dy)
#    print('plot')
#    scipy.misc.imsave('tabz.jpg', tabz)
#    im1 = plt.matshow(tabz)
#    plt.colorbar(im1,label='with summit')
#    plt.show 

    tabz1=tabz+repti(tabc,dx,dy)
#    scipy.misc.imsave('tabz1.jpg', tabz1)
#    im2 = plt.matshow(tabz1)
#    plt.colorbar(im2,label='with summit and in between')

    tabz2=tabz1+reptf(tabz1,tabc,dx,dy)
#    scipy.misc.imsave('tabz2.jpg', tabz2)
#    im3 = plt.matshow(tabz2)
#    plt.colorbar(im3,label='with full fill')
     
    tabz3=reptfc (tabz2,dx,dy)
#    scipy.misc.imsave('tabz3.jpg', tabz3)
#    im4 = plt.matshow(tabz3)
#    plt.colorbar(im4,label='with correct fill')
#    plt.show 
#    print(tabz3)
#  
    return tabz3

def pavs (tabc,tab,dx,dy,px,py,namedirtopcf,jpegpath,patchpath,thr,\
    iln,f,label,loca,typei,errorfile):
    """ generate patches from ROI"""
    #namedirtopcf = final/ILD_DB_txtROIs/35
    #jpegpath = final/jpeg
    #patchpath = final/jpeg
    #coefi=0.79296875
   #iln=slice2micronodulesdiffuse
    #label=micronodules
   #loca=diffuse
    #typei = bmp
#    print('enter pavs')

    mintaby= min(511,tabc.take([1],axis=1).min())
    maxtaby= min(511,tabc.take([1],axis=1).max())
    mintabx= min(511,tabc.take([0],axis=1).min())
    maxtabx= min(511,tabc.take([0],axis=1).max())
#    print(mintabx,maxtabx,mintaby,maxtaby)
    patchpathc=os.path.join(namedirtopcf,typei)
#    print patchpathc
    contenujpg = os.listdir(patchpathc)
    #contenujpg in  final/ILD_DB_txtROIs/35/jpg
    debnumslice=iln.find('_')+1
    endnumslice=iln.find('_',debnumslice)
    slicenumber=iln[debnumslice:endnumslice]
    if int(slicenumber)<10:
        slns='000'+slicenumber+'.'+typei
    elif int(slicenumber)<100:
        slns='00'+slicenumber+'.'+typei
    elif int(slicenumber)<1000:
          slns='0'+slicenumber+'.'+typei
    elif int(slicenumber)<10000:
          slns=slicenumber+'.'+typei
    tabp = np.zeros((dx, dy), dtype='i')
    tabf = np.copy(tab)
    pxy=float(px*py)
    i=max(mintabx-px,0)
    nbp=0
    strpac=''
    mini=min(maxtabx-px,dx-px)
    minj=min(maxtaby-py,dy-py)
    maxj=max(mintaby-py,0)
    errorliststring=[]
    for  n in contenujpg:
#                    print (area,n,i,j)
#                    print(slns)
        if n.find(slns)>0:
            orig = Image.open(namedirtopcf+'/'+typei+'/'+n)
            while i <= mini:
                j=maxj
                while j<=minj:
        #            print(i,j)
                    ii=0
                    area=0
                    while ii < px:
                        jj=0
                        while jj <py:
                            if tabf[j+jj][i+ii] >0:
                                area+=1
                            jj+=1
                        ii+=1
          
                    if float(area)/pxy >thr:
                        #good patch
#                   
                        crorig = orig.crop((i, j, i+px, j+py))
                         #detect black pixels
                         #imagemax=(crorig.getextrema())
                        imagemax=crorig.getbbox()
#                         print (imagemax)
                        if imagemax==None:

                            errortext='black pixel in: '+ f+' '+ iln+'\n'
                            if errortext not in errorliststring:
                                errorliststring.append(errortext)
                                print(errortext)
#                          
                        else:
                            nbp+=1
                            crorig.save(patchpath+'/'+label+'/'+loca+'/'+f+\
                             '_'+iln+'_'+str(nbp)+'.'+typei)
                        
                        
                        #                print('pavage',i,j)
#                        if gp:   
                            strpac=strpac+str(i)+' '+str(j)+'\n'
                            x=0
                            #we draw the rectange
                            while x < px:
                                y=0
                                while y < py:
                                    tabp[y+j][x+i]=4
                                    if x == 0 or x == px-1 :
                                        y+=1
                                    else:
                                        y+=py-1
                                x+=1
                            #we cancel the source
                            x=0
                            while x < px:
                                y=0
                                while y < py:
                                    tabf[y+j][x+i]=0
                                    y+=1
                                x+=1
                           
                    j+=1
                i+=1
            break
    
    else:
        print('ERROR image not found '+namedirtopcf+'/'+typei+'/'+n)
        errorfile.write('ERROR image not found '+namedirtopcf+\
        '/'+typei+'/'+n+'\n')#####
    tabp =tab+tabp
    mf=open(jpegpath+'/'+f+'_'+iln+'.txt',"w")
    mf.write('#number of patches: '+str(nbp)+'\n'+strpac)
    mf.close()
    scipy.misc.imsave(jpegpath+'/'+f+'_'+iln+'.jpg', tabp)
    if len(errorliststring) >0:
        for l in errorliststring:
            errorfile.write(l)
#    im = plt.matshow(tabp)
#    plt.colorbar(im,label='with pavage')
##    im = plt.matshow(tabf)
##    plt.colorbar(im,label='ff')
#    plt.show
#    print('fin')
    return nbp,tabp


def fileext(namefile,curdir,patchpath):
    listlabel=[]
    ofi = open(namefile, 'r')
    t = ofi.read()
    #print( t)
    ofi.close()
#
    nslice = t.count('slice')
#    print('number of slice:',nslice)
    numbercon = t.count('contour')
    nset=0
#    print('number of countour:',numbercon)
    spapos=t.find('SpacingX')
    coefposend=t.find('\n',spapos)
    coefposdeb = t.find(' ',spapos)
    coef=t[coefposdeb:coefposend]
    coefi=float(coef)
#    print('coef',coefi)
#    
    labpos=t.find('label')
    while (labpos !=-1):
#        print('boucle label')
        labposend=t.find('\n',labpos)
        labposdeb = t.find(' ',labpos)
        
        label=t[labposdeb:labposend].strip()
        if label.find('/')>0:
            label=label.replace('/','_')
    
#        print('label',label)
        locapos=t.find('loca',labpos)
        locaposend=t.find('\n',locapos)
        locaposdeb = t.find(' ',locapos)
        loca=t[locaposdeb:locaposend].strip()
#        print(loca)
 
        if loca.find('/')>0:
#            print('slash',loca)
            loca=loca.replace('/','_')
#            print('after',loca)
    
#        print('label',label)
#        print('localisation',loca)
        if label not in listlabel:
                    plab=os.path.join(patchpath,label)
                    ploc=os.path.join(plab,loca) 
                    listlabel.append(label+'_'+loca)     
                    listlabeld=os.listdir(patchpath)
                    if label not in listlabeld:
                            os.mkdir(plab)
                    listlocad=os.listdir(plab)
                    if loca not in listlocad:
                            os.mkdir(ploc)

        condslap=True
        slapos=t.find('slice',labpos)
        while (condslap==True):
#            print('boucle slice')

            slaposend=t.find('\n',slapos)
            slaposdeb=t.find(' ',slapos)
            slice=t[slaposdeb:slaposend].strip()
#            print('slice:',slice)

            nbpoint=0
            nbppos=t.find('nb_point',slapos)     
            conend=True
            while (conend):
                nset=nset+1
                nbpoint=nbpoint+1
                nbposend=t.find('\n',nbppos)
                tabposdeb=nbposend+1
                
                slaposnext=t.find('slice',slapos+1)
                nbpposnext=t.find('nb_point',nbppos+1)
                labposnext=t.find('label',labpos+1)
                #last contour in file
                if nbpposnext==-1:
                    tabposend=len(t)-1
                else:
                    tabposend=nbpposnext-1
                #minimum between next contour and next slice
                if (slaposnext >0  and nbpposnext >0):
                     tabposend=min(nbpposnext,slaposnext)-1 
                #minimum between next contour and next label
                if (labposnext>0 and labposnext<nbpposnext):
                    tabposend=labposnext-1
#                    
#                if (int(slice)==19):
#                    print(slapos,slaposnext,nbpposnext,labposnext,\
#                    tabposdeb,tabposend)
        #        print('fin tableau:',tabposend)
                nametab=curdir+'/patchfile/slice_'+str(slice)+'_'+str(label)+\
                '_'+str(loca)+'_'+str(nbpoint)+'.txt'
    #            print(nametab)
#
                mf=open(nametab,"w")
                mf.write('#label: '+label+'\n')
                mf.write('#localisation: '+loca+'\n')
                mf.write(t[tabposdeb:tabposend])
                mf.close()
                nbppos=nbpposnext 
                #condition of loop contour
                if (slaposnext >1 and slaposnext <nbpposnext) or\
                   (labposnext >1 and labposnext <nbpposnext) or\
                   nbpposnext ==-1:
                    conend=False
            slapos=t.find('slice',slapos+1)
            labposnext=t.find('label',labpos+1)
            #condition of loop slice
            if slapos ==-1 or\
            (labposnext >1 and labposnext < slapos ):
                condslap = False
        labpos=t.find('label',labpos+1)
#    print('total number of contour',nset,'in:' , namefile)
    return(listlabel,coefi)


mf.write('label  _  localisation\n')
mf.write('======================\n')

npat=0
for f in listdirc:
    #f = 35
    print('work on:',f)
    mf.write(f+'\n')
    nbpf=0
    posp=f.find('.',0)
    posu=f.find('_',0)
    namedirtopcf=namedirtopc+'/'+f

    if os.path.isdir(namedirtopcf):
        genebmp(namedirtopcf)
    remove_folder(namedirtopcf+'/patchfile')
    #namedirtopcf = final/ILD_DB_txtROIs/35
    if posp==-1 and posu==-1:
        contenudir = os.listdir(namedirtopcf)
#        print(contenudir)
     
        if  'patchfile' not in contenudir:
            os.mkdir(namedirtopcf+'/patchfile')
        fif=False
        for f1 in contenudir:
            if f1.find('.txt') >0 and (f1.find('CT')==0 or \
             f1.find('Tho')==0):
                npat+=1
                fif=True
                fileList =f1
                ##f1 = CT-INSPIRIUM-1186.txt
                pathf1=namedirtopcf+'/'+fileList
                #pathf1=final/ILD_DB_txtROIs/35/CT-INSPIRIUM-1186.txt
             
                labell,coefi =fileext(pathf1,namedirtopcf,patchpath)
#                print(label,loca)
                for ff in labell:
#                    print ff
                    mf.write(str(ff)+'\n')
                mf.write('--------------------------------\n')
                break
        if not fif:
             print('ERROR: no ROI txt content file', f)
             errorfile.write('ERROR: no ROI txt content file in: '+ f+'\n')
        
        listslice= os.listdir(namedirtopcf+'/patchfile') 
#        print('listslice',listslice)
        listcore =[]
        for l in listslice:
#                print(pathl)
            il1=l.find('.',0)
            j=0
            while l.find('_',il1-j)!=-1:
                j-=1
            ilcore=l[0:il1-j-1]
            if ilcore not in listcore:
                listcore.append(ilcore)
    #pathl=final/ILD_DB_txtROIs/35/patchfile/slice_2_micronodulesdiffuse_1.txt
#        print('listcore',listcore)
        for c in listcore:
            
            ftab=True
            tabzc = np.zeros((dimtabx, dimtaby), dtype='i')
            for l in listslice:
#                print('l',l,'c:',c)
                if l.find(c,0)==0:
                    pathl=namedirtopcf+'/patchfile/'+l
                    tabcff = np.loadtxt(pathl,dtype='f')
                    ofile = open(pathl, 'r')
                    t = ofile.read()
                    #print( t)
                    ofile.close()
                    labpos=t.find('label')
                    labposend=t.find('\n',labpos)
                    labposdeb = t.find(' ',labpos)
                    label=t[labposdeb:labposend].strip()
                    locapos=t.find('local')
                    locaposend=t.find('\n',locapos)
                    locaposdeb = t.find(' ',locapos)
                    loca=t[locaposdeb:locaposend].strip()
#                print(label,loca)
                #print(tabcff,coefi)
                    tabccfi=tabcff/coefi
#                print(tabccfi)
                    tabc=tabccfi.astype(int)
                    
#                print(tabc)
                    print('generate tables from:',l,'in:', f)
                    tabz= reptfull(tabc,dimtabx,dimtaby)
                    tabzc=tabz+tabzc
                    if ftab:
                        reftab=tabc
                        ftab=False
                    
                    reftab=np.concatenate((reftab,tabc),axis=0)
                    
                    print('end create tables')
                    il=l.find('.',0)
                    iln=l[0:il]
#                    print iln

            print('creates patches from:',iln, 'in:', f)
            nbp,tabz1=pavs (reftab,tabzc,dimtabx,dimtaby,dimpavx,dimpavy,namedirtopcf,\
                jpegpath, patchpath,thrpatch,iln,f,label,loca,typei,errorfile)
            print('end create patches')
            nbpf=nbpf+nbp
#    print(f,nbpf)
    ofilepw = open(jpegpath+'/nbpat_'+f+'.txt', 'w')
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
    
    
#################################################################    
#   calculate number of patches
contenupatcht = os.listdir(jpegpath) 
#        print(contenupatcht)
npatcht=0
for npp in contenupatcht:
#    print('1',npp)
    if npp.find('.txt')>0 and npp.find('nbp')==0:
#        print('2',npp)
        ofilep = open(jpegpath+'/'+npp, 'r')
        tp = ofilep.read()
#        print( tp)
        ofilep.close()
        numpos2=tp.find('number')
        numposend2=len(tp)
        #tp.find('\n',numpos2)
        numposdeb2 = tp.find(':',numpos2)
        nump2=tp[numposdeb2+1:numposend2].strip()
#        print(nump2)
        numpn2=int(nump2)
        npatcht=npatcht+numpn2
#        print(npatch)
ofilepwt = open(jpegpath+'/totalnbpat.txt', 'w')
ofilepwt.write('number of patches: '+str(npatcht))
ofilepwt.close()
mf.write('================================\n')
mf.write('number of datasets:'+str(npat)+'\n')
mf.close()
#################################################################
#data statistics on paches
#nametopc=os.path.join(cwd,namedirtop)
dirlabel=os.walk( patchpath).next()[1]
#file for data pn patches
filepwt = open(namedirtopc+'totalnbpat.txt', 'w')
ntot=0;

labellist=[]
localist=[]

for dirnam in dirlabel:
    dirloca=os.path.join(patchpath,dirnam)
#    print ('dirloca', dirloca)
    listdirloca=os.listdir(dirloca)
    label=dirnam
#    print ('dirname', dirname)

    loca=''
    if dirnam not in labellist:
            labellist.append(dirnam)
#    print('label:',label)
    for dlo in listdirloca:
        loca=dlo
        if dlo not in localist:      
            localist.append(dlo)
#        print('localisation:',loca)
        if label=='' or loca =='':
            print('not found:',dirnam)        
        subdir = os.path.join(dirloca,loca)
#    print(subdir)
        n=0
        listcwd=os.listdir(subdir)
        for ff in listcwd:
            if ff.find(typei) >0 :
                n+=1
                ntot+=1
#        print(label,loca,n) 
        filepwt.write('label: '+label+' localisation: '+loca+\
        ' number of patches: '+str(n)+'\n')
filepwt.close() 
##########################################################
errorfile.write('completed')
errorfile.close()
print('completed')