import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc
def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
###############################################################################################
def FindIntersection(InDir,SubDir, NewDir):
    for DirName in os.listdir(InDir):
        DirName=InDir+DirName
        if os.path.exists(DirName+"/"+SubDir):
            os.rename(DirName+"/"+SubDir,DirName+"/"+NewDir)
        else:
            print(DirName)


InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
SubDir=r"Vessel"
NewDir=r"Vessel"

FindIntersection(InDir,SubDir, NewDir)
