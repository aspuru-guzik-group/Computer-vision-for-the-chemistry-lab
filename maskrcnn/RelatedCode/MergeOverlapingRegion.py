import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc
# Find segments wich are the same segments (high overlap) and merge
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
         for  i in range(l):
             path1=SgDir+"/"+listfile[i]
             if not os.path.exists(path1):continue
             sg1 = cv2.imread(path1,0)
             CatName=listfile[i][listfile[i].find("Class__")+7:listfile[i].find("__ClasID__")]
             CatID=listfile[i][listfile[i].find("ClasID__")+8:listfile[i].find(".png")]
             for f in range(i+1,l):
                 path2 = SgDir + "/" + listfile[f]
                 if not os.path.exists(path2): continue
                 sg2 = cv2.imread(path2, 0)
                 if (sg1*sg2).sum()/(np.max([sg1.sum(),sg2.sum()]))>0.975:
                     CatName2 = listfile[f][listfile[f].find("Class__") + 7:listfile[f].find("__ClasID__")]
                     CatID2 = listfile[f][listfile[f].find("ClasID__") + 8:listfile[f].find(".png")]
                     CatName+="_"+CatName2
                     CatID+="_"+CatID2
                     # cv2.imshow(CatName,np.concatenate([sg1,sg2],axis=1)*200)
                     # cv2.waitKey()
                     # cv2.destroyAllWindows()
                     os.remove(path2)
             k+=1
             os.remove(path1)
             path = SgDir + "/" + str(k) + "_Class_" + CatName + "_CatID_" + CatID + ".png"
             print(path)
             cv2.imwrite(path, sg1)
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
SubDir=r"Vessel"
MergeOverLapping(InDir, SubDir)