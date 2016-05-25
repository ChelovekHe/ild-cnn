# coding: utf-8
#Sylvain Kritter 23 mai 2016
"""generate  bmp files from dicom files"""
import g
import dicom
import os
import scipy.misc



for dirName in g.patient_list:
    print ('generate  bmp files from dicom files in :',dirName)
    dirNameDir=os.path.join(g.path_patient,dirName)
    #directory for patches
    bmp_dir = os.path.join(dirNameDir, g.scanbmp)
    g.remove_folder(bmp_dir)    
    os.mkdir(bmp_dir)
    
    #list dcm files
    fileList = os.listdir(dirNameDir)

    for filename in fileList:
#        print(filename)
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            FilesDCM =(os.path.join(dirNameDir,filename))  
#           
            ds = dicom.read_file(FilesDCM)
            endnumslice=filename.find('.dcm')
            imgcore=filename[0:endnumslice]+'.'+g.typei
#            print imgcore
            bmpfile=os.path.join(bmp_dir,imgcore)
            scipy.misc.imsave(bmpfile, ds.pixel_array)
            
        #chek if lung mask present       
        if g.lungmask == filename:
         
             lung_dir = os.path.join(dirNameDir, g.lungmask)
             lung_bmp_dir = os.path.join(lung_dir,g.lungmaskbmp)
             lunglist = os.listdir(lung_dir)
             g.remove_folder(lung_bmp_dir)
#             if lungmaskbmp not in lunglist:
             os.mkdir(lung_bmp_dir)
#             print(lung_bmp_dir)
             for lungfile in lunglist:
#                print(lungfile)
                if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                    lungDCM =os.path.join(lung_dir,lungfile)  
                    dslung = dicom.read_file(lungDCM)
                    endnumslice=lungfile.find('.dcm')
                    lungcore=lungfile[0:endnumslice]+'.'+g.typei
                    lungcoref=os.path.join(lung_bmp_dir,lungcore)
                    scipy.misc.imsave(lungcoref, dslung.pixel_array)


