import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc

###############################################################################################
def MergeOverLapping(InDir,SubDir):
    for DirName in os.listdir(InDir):
         DirName=InDir+"//"+DirName
         SgDir=DirName+"/"+SubDir+"//"
         if not os.path.isdir(SgDir):
                      print(SgDir)
                      continue

         listfile=[]
         for fl in os.listdir(SgDir):
             if ".png" in fl:
                 listfile.append(fl)

         l=len(listfile)
         k=0
         im=cv2.imread(DirName+"//Image.png")
         for  i in range(l):
             path1=SgDir+"/"+listfile[i]
             if not os.path.exists(path1):continue
             sg1 = cv2.imread(path1,0)
             im[:,:,0]*=1-sg1
             im[:, :, 1] *= 1 - sg1
             cv2.imshow(listfile[i],im)
             cv2.waitKey()
             cv2.destroyAllWindows()


 #####################################################################3333

             # SG = cv2.imread(path,0)
             # Img = cv2.imread(ImFolder + ImName)
             # Img[:, :, 2] *= 1 - SG
             # Img[:, :, 1] *= 1 - SG
             # Img2 = cv2.imread(ImFolder + ImName)
             # Img=np.concatenate([Img,Img2],axis=1)
             # Im=cv2.resize(Img,(1000,500))
             # cv2.imshow(path,Im)
             # cv2.waitKey()
             # cv2.destroyAllWindows()
#########################################################################################################################

###########################################################################################################################
InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\\"
SubDir=r"Material"
MergeOverLapping(InDir, SubDir)