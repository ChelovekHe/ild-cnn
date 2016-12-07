# coding: utf-8
#Sylvain Kritter 21 septembre 2016
""" merge chu Grenoble and HUg patchdatabases"""
import os
import shutil

#global environment

source='CHU'
patchsource='TOPPATCH_16_set0_gci'
patsonorm='patches_norm'
dest='HUG'
patchdestT='TOPPATCH_16_set0_gci'
patchdest='patches_norm'


#########################################################################
cwd=os.getcwd()
(top,tail)=os.path.split(cwd)
sourcedir = os.path.join(top,source)
sourcedir = os.path.join(sourcedir,patchsource)
sourcedir = os.path.join(sourcedir,patsonorm)

destdir=os.path.join(top,dest)
destdir1=os.path.join(destdir,patchdestT)
destdir=os.path.join(destdir1,patchdest)

print sourcedir,destdir

listsource=os.listdir(sourcedir)
print listsource
for l in listsource:
    print l
    sourcedirl = os.path.join(sourcedir,l)
    listsourcel=os.listdir(sourcedirl)
    for k in listsourcel:
      print k
      sourcedirk = os.path.join(sourcedirl,k)
      desdirl = os.path.join(destdir,l)
      destdirk = os.path.join(desdirl,k)
#      print l, k,sourcedirk,destdirk
      if not os.path.exists(destdirk):          
          os.mkdir(destdirk)     
      listsourcem=os.listdir(sourcedirk)
      for f in listsourcem:
        sourcedirf = os.path.join(sourcedirk,f)
        sourcedestf = os.path.join(destdirk,f)
#        print sourcedirf,sourcedestf

        shutil.copyfile(sourcedirf,sourcedestf)
    

#    