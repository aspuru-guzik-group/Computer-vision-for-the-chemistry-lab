import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc
# Create semantic map from instance map
#############################################################################################
def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()

###############################################################################################
def GenerateSemanticMap(InDir,SubDir):
    ppp=0
    for DirName in os.listdir(InDir):
         print(DirName)
         ppp+=1
         print(ppp)
         pig = False
         DirName=InDir+"//"+DirName

         SemDir=DirName+"//Semantic//"
         Im = cv2.imread(DirName + "/Image.png")

         for p in range(4):
             SgDir = DirName + "/" + SubDir[p] + "//"
             if not os.path.exists(SgDir): continue
             for name in os.listdir(SgDir):
                 path1 = SgDir + "/" + name
                 if not os.path.exists(path1): continue
                 sg = cv2.imread(path1)
                 sg[:,:,1]*=0
                 sg[:, :, 2] *= 0
                 sg[sg>2] = 0
                 I1 = Im.copy()
                 if np.ndim(sg)==2:
                         I1[:, :, 0] *= 1 - sg
                         I1[:, :, 1] *= 1 - sg
                         I1 = np.concatenate([Im, I1], axis=1)
                 else:
                     I1=(I1/3+sg*50).astype(np.uint8)
                     I1=np.concatenate([Im,I1,(sg*70).astype(np.uint8)],axis=1)
                 print(path1)
                 #show(I1)
                 cv2.imwrite(path1,I1)
             os.rename(SgDir,SgDir.replace(SubDir[p],SubDir[p]+"V"))
####################################################################################################
InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
SubDir=[r"Semantic","Material",r"Parts",r"Vessel"]



GenerateSemanticMap(InDir,SubDir)