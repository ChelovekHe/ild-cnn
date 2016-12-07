# coding: utf-8
import os
from scipy import misc
import numpy as np
#data set creation, 6 enhavcement,  equalisation for no-lung only

import cPickle as pickle
from sklearn.cross_validation import train_test_split
import random
from random import randrange
import cv2

namedirHUG = 'HUG'
namedirCHU = 'CHU'
#define the working directory
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
path_HUG=os.path.join(cwdtop,namedirHUG)
path_CHU=os.path.join(cwdtop,namedirCHU)
patch_dir1=os.path.join(path_HUG,'patches_15_set4_g')
#patch_dir1=os.path.join(path_HUG,'patches_essai')
subsample=10
#print patch_dir1
patch_dir2=os.path.join(path_CHU,'patches_15_set4_g')
#patch_dir2=os.path.join(path_CHU,'patches_essai')

pickle_dir=os.path.join(cwdtop,'pickle_ds15')
if not os.path.exists(pickle_dir):
   os.mkdir(pickle_dir)  

patchdirset=(patch_dir1,patch_dir2)
#patchdirset=(patch_dir1,)
#print patchdirset
#define a list of used labels
nosubsample=['chupr',]
usedclassif = ['nolung',  'lung']
#print (usedclassif)
classif ={
'nolung':0,
'lung':1
}
#augmentation factor
augf=6
#define a dictionary with labels
#classif={}
#i=0
for f in usedclassif:
    print (f, classif[f])
    
#define another dictionaries to calculate the number of label
classNumberInit={}
for f in usedclassif:
    classNumberInit[f]=0

classNumberNew={}
for f in usedclassif:
    classNumberNew[f]=0

#another to define the coeff to apply
classConso={}

print ('classification used:',classif)
print '----------------'
# list all directories under patch directory. They are representing the categories

# print what we have as categories
for p in patchdirset:
#    print p
    category_list=os.listdir( p)
    print (p,' ','all classes:',category_list)
print '----------------'

# go through all categories to calculate the number of patches per class
# 
for category in usedclassif:
    for p in patchdirset:
        category_dir = os.path.join(p, category)
        ldtdir= os.listdir(category_dir)
        for f in ldtdir:
                dirf=os.path.join(category_dir,f)
#                image_files = (os.listdir(dirf))
                image_files = [name for name in os.listdir(dirf) if name.find('.bmp')>0]
                for fil in image_files:
#                    if fil.find('.bmp') > 0:
                        classNumberInit[category]=classNumberInit[category]+augf


total=0
print('number of patches init')
for f in usedclassif:
    print (f,classNumberInit[f],classNumberInit[f]/augf)
    total=total+classNumberInit[f]
print('total:',total,total/augf)
print '----------------'

#define coeff
maxl=0
for f in usedclassif:
   if classNumberInit[f]>maxl and f !='nolung':
      maxl=classNumberInit[f]
print ('max number of patches in : ',maxl)
print '----------------'
for f in usedclassif:
  classConso[f]=float(maxl)/classNumberInit[f]
for f in usedclassif:
    print (f,' {0:.2f}'.format (classConso[f]))
print '----------------'



def listcl(lc):
    dlist = []
    print('list patches from class : ',lc)
#    while classNumberNew[lc]<m:    
    for p in patchdirset:
        category_dir = os.path.join(p, lc)
        print category_dir
        ldtdir= os.listdir(category_dir)
        for f in ldtdir:
                dirf=os.path.join(category_dir,f)
#                image_files1 = (os.listdir(dirf))
                image_files1 = [name for name in os.listdir(dirf) if name.find('.bmp')>0]
                print  'the path into the sub categories is: ', dirf
        #print subCategory_dir
#                image_files = (os.listdir(category_dir))


                if f in nosubsample:
                    print f,' no subsample'

                    image_files=image_files1
                else:
                    print f,' subsample'
                    cnif=int(len(image_files1)/subsample)
                    image_files = random.sample(image_files1,cnif)
                    
                print len(image_files1),len(image_files)
                for filei in image_files:
            
#                    if filei.find('.bmp') > 0:
                    # load the .bmp file into array
                        classNumberNew[lc]=classNumberNew[lc]+augf
            #            image = misc.imread(os.path.join(category_dir,filei), flatten= 0)
                        image = cv2.imread(os.path.join(dirf,filei),0)          
                        # 1 append the array to the dataset list                        
                        dlist.append(image)
            ##                        #2 created rotated copies of images
                        image90 = np.rot90(image)                        
                        dlist.append(image90)
            ##                        #3 created rotated copies of images                        
                        image180 = np.rot90(image90)
                        dlist.append(image180)
            ##                        #4 created rotated copies of images                                          
                        image270 = np.rot90(image180)                      
                        dlist.append(image270)   
                    
                    # 5flip fimage left-right
                        imagefliplr=np.fliplr(image)
                        dlist.append(imagefliplr)            
                        
                    # 6 flip fimage up-down           
                        imageflipud=np.flipud(image)                           
                        dlist.append(imageflipud)
            
    return dlist


# list for the merged pixel data

# list of the label data
label_list = []
dataset_list =[]
for f in usedclassif:
     print('work on :',f)
    #fill list with patches
     resul = listcl(f)
#     print 'resul:', resul
#     resul=equal(maxl,dlf,f)
     i=0
     while i <  classNumberNew[f]:
#        print resul[i], classif[f]
        dataset_list.append(resul[i])
        label_list.append(classif[f])
        i+=1
 

print '----------------'
print 'number of patches:'
for f in usedclassif:
    print ('init',f,classNumberInit[f])
    print ('after',f,classNumberNew[f])
    print '----------------'
print '----------------'


print (len(dataset_list),len(label_list))


# transform dataset list into numpy array                   
X = np.array(dataset_list)
#this is already in greyscale
# use only one of the 3 color channels as greyscale info
#X = dataset[:,:, :,1]

print 'dataset shape is now: ', X.shape
#print('X22 as example:', X[22])
# 
y = np.array(label_list)
# sampling item 22
#print ('y22 as example:',y[22])
print '----------------'
print ('Xshape : ',X.shape)
print ('yshape : ',y.shape)

print '----------------'
X_train, X_intermediate, y_train, y_intermediate = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.5, random_state=42)
print ('Xtrain :',X_train.shape)
print ('Xval : ',X_val.shape)
print ('Xtest : ',X_test.shape)
print ('ytrain : ',y_train.shape)
print ('ytest : ',y_test.shape)
print ('yval : ',y_val.shape)
    

# save the dataset and label set into serial formatted pkl 

pickle.dump(X_train, open( os.path.join(pickle_dir,"X_train.pkl"), "wb" ))
pickle.dump(X_test, open( os.path.join(pickle_dir,"X_test.pkl"), "wb" ))
pickle.dump(X_val, open(os.path.join(pickle_dir,"X_val.pkl"), "wb" ))
pickle.dump(y_train, open( os.path.join(pickle_dir,"y_train.pkl"), "wb" ))
pickle.dump(y_test, open( os.path.join(pickle_dir,"y_test.pkl"), "wb" ))
pickle.dump(y_val, open( os.path.join(pickle_dir,"y_val.pkl"), "wb" ))

class_weights={}
for f in usedclassif:
#    print f,classNumberNewTr[f]
    class_weights[classif[f]]=round(float(classNumberNew['lung'])/classNumberNew[f],3)
print '---------------'

print 'weights'
print class_weights
pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ))
# testing if pickls was working fine
recuperated_X_train = pickle.load( open( os.path.join(pickle_dir,"X_train.pkl"), "rb" ) )


#print ('recuparated 22 as example:',recuperated_X_train[22])


