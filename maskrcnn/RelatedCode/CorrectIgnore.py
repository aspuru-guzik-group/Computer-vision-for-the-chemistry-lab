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
def AddIgnore(InDir):
    ooo=0
    for DirName in os.listdir(InDir):
         print(ooo)
         ooo+=1

         DirName=InDir+"//"+DirName
         Im = cv2.imread(DirName + "/Image.png")
         Ignore0 = cv2.imread(DirName + "/Ignore.png",0)
         Ignore=(Ignore0==3)
         if Ignore.sum()==0: continue
#=========================================================================
         Im[:, :, 0] *= 1 - Ignore.astype(np.uint8)
         Im[:, :, 1] *= 1 - Ignore.astype(np.uint8)
         cv2.imshow(" i (ignore) c(correct)", Im)

         while (True):
             ch = chr(cv2.waitKey())
             if ch == 'i' or ch == 'c': break

         cv2.destroyAllWindows()
         if ch == 'i': continue
         Ignore0[Ignore]=2
         cv2.imwrite(DirName + "/Ignore.png", Ignore0)
#=========================================================================


InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
AddIgnore(InDir)