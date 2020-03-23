# Copy Ignore to main folder

import numpy
import json
import cv2
import numpy as np
import os
from shutil import copyfile
import scipy.misc as misc
InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\MergedSetAll\Annotations\IgnoredNEW\\"
OutDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
for name in os.listdir(InDir):
    InPath=InDir+name
    OutPath=OutDir+"//"+name[:-4]+"//Ignore.png"
    copyfile(InPath,OutPath)
    print(InPath)

    print(OutPath)



#
# InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\\"
# SubDir=r"Material"
# FindIntersection(InDir, SubDir)