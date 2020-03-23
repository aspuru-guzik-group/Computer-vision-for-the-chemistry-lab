import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc
# Add Ignore to vessel
#############################################################################################
def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
###############################################################################################
def AddIgnore(InDir,SubDir):
    ooo=0
    for DirName in os.listdir(InDir):
         print(ooo)
         ooo+=1

         DirName=InDir+"//"+DirName
         Im = cv2.imread(DirName + "/Image.png")
         Ignore = cv2.imread(DirName + "/Ignore.png", 0)
         Ignore = (Ignore==3) + (Ignore==2)
         if Ignore.sum()==0: continue
         SgDir=DirName+"/"+SubDir+"//"
#=========================================================================
         # Im[:, :, 0] *= 1 - Ignore.astype(np.uint8)
         # Im[:, :, 1] *= 1 - Ignore.astype(np.uint8)
         # cv2.imshow(" i (ignore) a(add)", Im)
         #
         # while (True):
         #     ch = chr(cv2.waitKey())
         #     if ch == 'i' or ch == 'a': break
         #
         # cv2.destroyAllWindows()
         # if ch == 'i': continue
         SgDir=DirName+"/"+SubDir
#=========================================================================
         for  name in os.listdir(SgDir):
             path1=SgDir+"/"+name
             print(path1)
             if not os.path.exists(path1):continue
             sg1=cv2.imread(path1)
             sg1[Ignore,2]=7
             cv2.imwrite(path1, sg1)
             sg1 = cv2.imread(path1)
             show(sg1*100)


###########################################################################################################################
InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
SubDir=r"PartsVV"
AddIgnore(InDir,SubDir)