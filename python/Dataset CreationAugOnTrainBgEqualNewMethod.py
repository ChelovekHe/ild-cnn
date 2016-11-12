# coding: utf-8
'''create dataset from patches using split into 3 bmp databases and
 augmentation only on training, using bg, bg number limited to max and with equalization
 for all in training, validation and test
 defined used pattern set
 '''
import os
import dircache
import sys
import shutil
from scipy import misc
import numpy as np

import cPickle as pickle
#from sklearn.cross_validation import train_test_split
import random

#####################################################################
#define the working directory
HUG='CHU'   



#output pickle dir with dataset   
pickel_dirsource='pickle_ds24'


#input pickle dir with dataset to merge
pickel_dirsourceToMerge='pickle_ds22'


#input patch directory
patch_dirsource=os.path.join('TOPPATCH_16_set0','patches_norm')

#output database for database generation
patch_dirSplitsource=   'chu16set0'


#define the pattern set
pset=0

###############################################################
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)


pickle_dir=os.path.join(cwdtop,pickel_dirsource) 
pickle_dirToMerge=os.path.join(cwdtop,pickel_dirsourceToMerge)  


patch_dir=os.path.join(dirHUG,patch_dirsource)
print patch_dir


patch_dirSplit=os.path.join(dirHUG,patch_dirSplitsource)
patch_dir_Tr=os.path.join(patch_dirSplit,'p_Tr')
patch_dir_V=os.path.join(patch_dirSplit,'p_V')
patch_dir_Te=os.path.join(patch_dirSplit,'p_Te')


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

remove_folder(patch_dirSplit)
os.mkdir(patch_dirSplit)  
remove_folder(patch_dir_Tr)
os.mkdir(patch_dir_Tr)  
remove_folder(patch_dir_V)
os.mkdir(patch_dir_V)  
remove_folder(patch_dir_Te)
os.mkdir(patch_dir_Te)  
   

remove_folder(pickle_dir)
os.mkdir(pickle_dir)  



#define a list of used labels
if pset ==0:
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
        'bronchiectasis'
        ]
        
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
elif pset==1:
    usedclassif = [
        'back_ground',
        'consolidation',
        'ground_glass',
        'healthy'
    #    ,'cysts'
        ]
        
    classif ={
    'back_ground':0,
    'consolidation':1,
    'ground_glass':2,
    'healthy':3
    #,'cysts':4
    }
elif pset==2:
        usedclassif = [
        'back_ground',
        'fibrosis',
        'healthy',
        'micronodules'
        ,'reticulation'
        ]
        
        classif ={
    'back_ground':0,
    'fibrosis':1,
    'healthy':2,
    'micronodules':3,
    'reticulation':4,
    }
elif pset==3:
    usedclassif = [
        'back_ground',
        'healthy',
        'air_trapping',
        ]
    classif ={
        'back_ground':0,
        'healthy':1,
        'air_trapping':2,
        }
else:
        print 'eRROR :', pset, 'not allowed'



#print (usedclassif)

#augmentation factor
augf=6
#define a dictionary with labels
#classif={}
#i=0
print ('classification used:')
for f in usedclassif:
    print (f, classif[f])

print '----------'
#define another dictionaries to calculate the number of label
classNumberInit={}    
classNumberNewTr={}
classNumberNewV={}
classNumberNewTe={}
actualClasses=[]

for f in usedclassif:
    classNumberInit[f]=0
    classNumberNewTr[f]=0
    classNumberNewV[f]=0
    classNumberNewTe[f]=0

# list all directories under patch directory. They are representing the categories

category_list=os.walk( patch_dir).next()[1]


# print what we have as categories
print ('all actual classes:',category_list)
print '----------'

usedclassifFinal=[f for f in usedclassif if f in category_list]

# print what we have as categories and in used one
print ('all actual classes:',usedclassifFinal)
print '----------'
# go through all categories to calculate the number of patches per class
# 
for category in usedclassifFinal:
    category_dir = os.path.join(patch_dir, category)
    print  'the path into the categories is: ', category_dir
    sub_categories_dir_list = (os.listdir(category_dir))
    #print 'the sub categories are : ', sub_categories_dir_list
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
        print  'the path into the sub categories is: ',subCategory_dir
        #print subCategory_dir
        image_files = (os.listdir(subCategory_dir))
        
       
        for filei in image_files:
            
            if filei.find('.bmp') > 0:
                
                classNumberInit[category]=classNumberInit[category]+1


total=0
print('number of patches init')
for f in usedclassifFinal:
    print ('class:',f,classNumberInit[f])
    total=total+classNumberInit[f]
print('total:',total)
print '----------'

#define coeff
maxl=0
for f in usedclassifFinal:
   if classNumberInit[f]>maxl and f !='back_ground':
      maxl=classNumberInit[f]
print ('max number of patches in : ',maxl)
print '----------'
#artificially clamp back-ground to maxl
classNumberInit['back_ground']=maxl
classConso={}
for f in usedclassifFinal:
  classConso[f]=float(maxl)/classNumberInit[f]
for f in usedclassifFinal:
    print (f,' {0:.2f}'.format (classConso[f]))
print '----------'


def   createStruct(f):
    print('Create patches directories from class : ',f)
    
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
    remove_folder(patch_dir_Tr_f)
    os.mkdir(patch_dir_Tr_f) 

    patch_dir_V_f=os.path.join(patch_dir_V,f)
    remove_folder(patch_dir_V_f)
    os.mkdir(patch_dir_V_f) 

    patch_dir_Te_f=os.path.join(patch_dir_Te,f)
    remove_folder(patch_dir_Te_f)
    os.mkdir(patch_dir_Te_f) 
    
def  copypatch(f):
    cnif=maxl
    print('copy all in new training  directory for:',f)
    category_dir = os.path.join(patch_dir, f)
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
#            print  ('the path into the categories is: ', lc)
    sub_categories_dir_list = (os.listdir(category_dir))
    print ('the sub categories are : ', sub_categories_dir_list)
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
                #print  'the path into the sub categories is: '
                #print subCategory_dir
        if f !='back_ground':
            #copy all the files if not back-ground
            image_files = (os.listdir(subCategory_dir))
    
            for filei in image_files:
    
                if filei.find('.bmp') > 0:
                            fileSource=os.path.join(subCategory_dir, filei)
                            fileDest=os.path.join(patch_dir_Tr_f, filei)
                        # load the .bmp file into dest directory
                            shutil.copyfile(fileSource,fileDest)
        else:
             #copy only randomly maxl file for back-ground
            while cnif > 0:
                dircache.reset()
                filename = random.choice(dircache.listdir(subCategory_dir))
                fileSource=os.path.join(subCategory_dir, filename)
                fileDest=os.path.join(patch_dir_Tr_f, filename)
                shutil.copyfile(fileSource,fileDest)
                cnif-=1           

def  selectpatch(f,n,t):
    print(n,' patch selection for:',f, 'for:',t )
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
    cni=classNumberInit[f]
    cnif=cni*n
    if t=='T':
        dirdest=os.path.join(patch_dir_Te,f)
    else:
        dirdest=os.path.join(patch_dir_V,f)
    while cnif > 0:
        dircache.reset()
        filename = random.choice(dircache.listdir(patch_dir_Tr_f))
        fileSource=os.path.join(patch_dir_Tr_f, filename)
        fileDest=os.path.join(dirdest, filename)
        # load the .bmp file into dest directory
#        print fileSource
#        print fileDest
        shutil.move(fileSource,fileDest)
        cnif-=1
        
  

def listcl(f,p,m):
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
    patch_dir_V_f=os.path.join(patch_dir_V,f)
    patch_dir_Te_f=os.path.join(patch_dir_Te,f)
    dlist = []
    classNumberN=0
    loop=0
    if p=='Tr':
        act='Training Set'
        category_dir=patch_dir_Tr_f
        maxp=m*augf*0.5 #-3 to take into account rounding
    elif p=='V':
        act='Validation Set'
        category_dir=patch_dir_V_f
        maxp=m*0.25
    else:
        act='Test Set'
        category_dir=patch_dir_Te_f
        maxp=m*0.25
    print('list patches from class : ',f, act)                 
    
#    while classNumberNew[lc]<m:        
#    category_dir_f = os.path.join(category_dir, f)
#            print  ('the path into the categories is: ', lc)
                #print subCategory_dir
    image_files = os.listdir(category_dir)

    while classNumberN<maxp: 
        loop+=1
        print('loop;',loop,'for :',f,'classnumber:',classNumberN,'max:',maxp)
       
        for filei in image_files:
    
                        if filei.find('.bmp') > 0:
                            image = misc.imread(os.path.join(category_dir,filei), flatten= 0)
                        # load the .bmp file into array
                            if p=='Tr':                        
                                classNumberN=classNumberN+augf                         
                                #print image                  
                                # 1 append the array to the dataset list                        
                                dlist.append(image)
        #                        #2 created rotated copies of images
                                image90 = np.rot90(image)                        
                                dlist.append(image90)
        #                        #3 created rotated copies of images                        
                                image180 = np.rot90(image90)
                                dlist.append(image180)
        #                        #4 created rotated copies of images                                          
                                image270 = np.rot90(image180)                      
                                dlist.append(image270)   
                                
                                #5 flip fimage left-right
                                imagefliplr=np.fliplr(image)                  
                                dlist.append(imagefliplr)
                                
                                #6 flip fimage up-down
                                imageflipud=np.flipud(image)                                   
                                dlist.append(imageflipud)
                            else:
                                classNumberN=classNumberN+1
                                #print image                  
                                # 1 append the array to the dataset list                        
                                dlist.append(image)

                            
    return dlist,classNumberN



# main program 
label_listTr = []
label_listV = []
label_listTe = []
dataset_listTr =[]
dataset_listV =[]
dataset_listTe =[]
for f in usedclassifFinal:
     print('work on :',f)
     dataset_listTri =[]
     dataset_listVi =[]
     dataset_listTei =[]
     #create structure dir for f
     createStruct(f)
     #copy patches in new directory flat
     copypatch(f)
     # select 1/4 patches for test
     selectpatch(f,0.25,'T')
#     select 1/4 patches for validation
     selectpatch(f,0.25,'V')
     
    #fill list with patches

     dataset_listTri,classNumberNewTr[f] = listcl(f,'Tr',maxl)
     dataset_listVi ,classNumberNewV[f]= listcl(f,'V',maxl)
     dataset_listTei,classNumberNewTe[f] = listcl(f,'Te',maxl)
#     resul=equal(maxl,dlf,f)

     i=0
     while i <  classNumberNewTr[f] :
        dataset_listTr.append(dataset_listTri[i])
        label_listTr.append(classif[f])
        i+=1
     i=0
     while i <  classNumberNewV[f] :         
        dataset_listV.append(dataset_listVi[i])
        label_listV.append(classif[f])
        i+=1
     i=0
     while i <  classNumberNewTe[f]:
        dataset_listTe.append(dataset_listTei[i])
        label_listTe.append(classif[f])
        i+=1
print '---------------------------'
for f in usedclassifFinal:
    print ('init',f,classNumberInit[f])
    print ('after training',f,classNumberNewTr[f])
    print ('after validation',f,classNumberNewV[f])
    print ('after test',f,classNumberNewTe[f])
    print '---------------------------'
#    print ('final',f,classNumberFinal[f])

print '---------------------------'
print ('training set:',len(dataset_listTr),len(label_listTr))
print ('validation set:',len(dataset_listV),len(label_listV))
print ('test set:',len(dataset_listTe),len(label_listTe))
# transform dataset list into numpy array                   
X_train = np.array(dataset_listTr)
y_train = np.array(label_listTr)
X_val = np.array(dataset_listV)
y_val = np.array(label_listV)
X_test = np.array(dataset_listTe)
y_test = np.array(label_listTe)
#this is already in greyscale
# use only one of the 3 color channels as greyscale info
#X = dataset[:,:, :,1]

#print 'dataset shape is now: ', X.shape
#print('X22 as example:', X[22])
## 
#y = np.array(label_list)
## sampling item 22
#print ('y22 as example:',y[22])
#
#print ('Xshape : ',X.shape)
#print ('yshape : ',y.shape)
#
#
#X_train, X_intermediate, y_train, y_intermediate = train_test_split(X, y, test_size=0.5, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.5, random_state=42)
print '-----------INIT----------------'
print ('Xtrain :',X_train.shape)
print ('Xval : ',X_val.shape)
print ('Xtest : ',X_test.shape)
print ('ytrain : ',y_train.shape)
print ('yval : ',y_val.shape)
print ('ytest : ',y_test.shape)
    
#print ('22 as example:',X_train[22])
# save the dataset and label set into serial formatted pkl 
#
#pickle.dump(X_train, open( os.path.join(pickle_dir,"X_train.pkl"), "wb" ))
#pickle.dump(X_test, open( os.path.join(pickle_dir,"X_test.pkl"), "wb" ))
#pickle.dump(X_val, open(os.path.join(pickle_dir,"X_val.pkl"), "wb" ))
#pickle.dump(y_train, open( os.path.join(pickle_dir,"y_train.pkl"), "wb" ))
#pickle.dump(y_test, open( os.path.join(pickle_dir,"y_test.pkl"), "wb" ))
#pickle.dump(y_val, open( os.path.join(pickle_dir,"y_val.pkl"), "wb" ))


#load to merge pickle
#recuperated_X_train = pickle.load( open( os.path.join(pickle_dirToMerge,"X_train.pkl"), "rb" ) )
recuperated_X_train = pickle.load( open( os.path.join(pickle_dirToMerge,"X_train.pkl"), "rb" ) ).tolist()
recuperated_X_test = pickle.load( open( os.path.join(pickle_dirToMerge,"X_test.pkl"), "rb" ) ).tolist()
recuperated_X_val = pickle.load( open( os.path.join(pickle_dirToMerge,"X_val.pkl"), "rb" ) ).tolist()
recuperated_y_train = pickle.load( open( os.path.join(pickle_dirToMerge,"y_train.pkl"), "rb" ) ).tolist()
recuperated_y_test = pickle.load( open( os.path.join(pickle_dirToMerge,"y_test.pkl"), "rb" ) ).tolist()
recuperated_y_val = pickle.load( open( os.path.join(pickle_dirToMerge,"y_val.pkl"), "rb" ) ).tolist()
#print ('recuparated 22 as example:',recuperated_X_train[22])
print '-----------to merge----------------'
print ('XtrainRecup :',len (recuperated_X_train))
print ('XvalRecup : ',len(recuperated_X_val))
print ('XtestRecup : ',len(recuperated_X_test))
print ('ytrainRecup : ',len(recuperated_y_train))
print ('yvalRecup : ',len(recuperated_y_val))
print ('ytestRecup : ',len(recuperated_y_test))


X_train = np.array(dataset_listTr+recuperated_X_train)
X_val = np.array(dataset_listV+recuperated_X_val)
y_train = np.array(label_listTr+recuperated_y_train)
y_val = np.array(label_listV+recuperated_y_val)
X_test = np.array(dataset_listTe+recuperated_X_test)
y_test = np.array(label_listTe+recuperated_y_test)


print '-----------after merge----------------'
print ('Xtrain :',X_train.shape)
print ('Xval : ',X_val.shape)
print ('Xtest : ',X_test.shape)
print ('ytrain : ',y_train.shape)
print ('yval : ',y_val.shape)
print ('ytest : ',y_test.shape)
    
pickle.dump(X_train, open( os.path.join(pickle_dir,"X_train.pkl"), "wb" ))
pickle.dump(X_test, open( os.path.join(pickle_dir,"X_test.pkl"), "wb" ))
pickle.dump(X_val, open(os.path.join(pickle_dir,"X_val.pkl"), "wb" ))
pickle.dump(y_train, open( os.path.join(pickle_dir,"y_train.pkl"), "wb" ))
pickle.dump(y_test, open( os.path.join(pickle_dir,"y_test.pkl"), "wb" ))
pickle.dump(y_val, open( os.path.join(pickle_dir,"y_val.pkl"), "wb" ))