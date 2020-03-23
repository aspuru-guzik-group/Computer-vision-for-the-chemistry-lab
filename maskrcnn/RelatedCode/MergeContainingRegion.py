import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc
# If Material region contain other ofer to transefer the first material class to the second
###############################################################################################
def MergeContainingRegion(InDir,SubDir):
    ooo=0
    for DirName in os.listdir(InDir):
         print(ooo)
         ooo+=1

         DirName=InDir+"//"+DirName
         Im = cv2.imread(DirName + "/Image.png")
         SgDir=DirName+"/"+SubDir+"//"
         if not os.path.isdir(SgDir):
                      print(SgDir + "NOT EXISTS")
                      continue

         listfile=[]
         for fl in os.listdir(SgDir):
             if ".png" in fl:
                 listfile.append(fl)

         l=len(listfile)
         k=0
         for  i in range(l):

             path1=SgDir+"/"+listfile[i]
             print(path1)
             if not os.path.exists(path1):continue
             sg1=cv2.imread(path1)[:, :, 0]
             sg1 = ((sg1==1) + (sg1==2)).astype(np.uint8)
             CatName=listfile[i][listfile[i].find("Class_")+6:listfile[i].find("_CatID_")]
             CatID=listfile[i][listfile[i].find("CatID_")+6:listfile[i].find(".png")]
             for f in range(l):
                 if f==i: continue
                 path2 = SgDir + "/" + listfile[f]
                 if not os.path.exists(path2): continue
                 sg2 = (cv2.imread(path2)[:,:,0]>0).astype(np.uint8)
                 if (sg1*sg2).sum()/(np.max([sg1.sum()]))>0.7:
                     CatName2 = listfile[f][listfile[f].find("Class_") + 6:listfile[f].find("_CatID_")]
                     CatID2 = listfile[f][listfile[f].find("CatID_") + 6:listfile[f].find(".png")]
                     if CatName2 in CatName: continue

                     # ..................................................................
                     Txt = CatName+"<-- s(skip)  m(merge) -->"+CatName2
                     Im1 = Im.copy()
                     Im1[:, :, 0] *= 1 - sg1
                     Im1[:, :, 2] *= 1 - sg2

                     cv2.imshow(Txt + "2", cv2.resize(Im1, (700, 700)))
                     cv2.imshow(Txt, cv2.resize(np.concatenate([sg1, sg2], axis=1) * 250, (1000, 500)))

                     while (True):
                         ch = chr(cv2.waitKey())
                         if ch == 's' or ch == 'm': break

                     cv2.destroyAllWindows()
                     if ch == 'm':
                         CatName += "_" + CatName2
                         CatID += "_" + CatID2
                     #........................................................................................
                     #..........................................................................................
                     # cv2.imshow(CatName,np.concatenate([sg1,sg2],axis=1)*200)
                     # cv2.waitKey()
                     # cv2.destroyAllWindows()

             k+=1
             Nm=listfile[i][:listfile[i].find("_")]
             pathNEW = SgDir + "/" + Nm + "_Class_" + CatName + "_CatID_" + CatID + ".png"
             if not pathNEW==path1:
                 print(pathNEW)
                 os.rename(path1, pathNEW)
                 listfile[i] = str(k) + "_Class_" + CatName + "_CatID_" + CatID + ".png"
                 print(pathNEW)
                 if not os.path.exists(pathNEW) or os.path.exists(path1):
                     print("ERRROOR")
                     exit(0)
         os.rename(SgDir, SgDir.replace(SubDir, "MaterialVVX"))


          ###   cv2.imwrite(path, sg1)
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
InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
SubDir=r"MaterialVVXX"
MergeContainingRegion(InDir,SubDir)