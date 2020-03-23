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
##############################################################################################
Cat={}
Cat[1]='Vessel'
Cat[2]='V_Label'
Cat[3]='V_Cork'
Cat[4]='V_Parts_GENERAL'
Cat[5]='Ignore'
Cat[6]='Liquid_GENERAL'
Cat[7]='Liquid Suspension'
Cat[8]='Foam'
Cat[9]='Gel'
Cat[10]='Solid_GENERAL'
Cat[11]='Granular'
Cat[12]='Powder'
Cat[13]='Solid Bulk'
Cat[14]='Vapor'
Cat[15]='Other Material'
Cat[16]='Filled'

CatLiquid=6
CatSolid=10
CatFilled=16
CatVParts=4
SolidLabels={9,10,11,12,13}
LiquidLabels={6,7,9}
FilledLabels={6,7,8,9,10,11,12,13,15}
PartsLabels={2,3,4}
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
         Ignore = cv2.imread(DirName + "/Ignore.png", 0)
         SegMaps=np.zeros([17,Im.shape[0],Im.shape[1],3],dtype=np.uint8)
         SegMaps[5,:,:,0][(Ignore>0)*(Ignore<4)]=1
         IgnoreIf=(Ignore==4)
         if os.path.exists(SemDir):continue

         for p in range(3):
             SgDir = DirName + "/" + SubDir[p] + "//"

             for name in os.listdir(SgDir):
                 path1 = SgDir + "/" + name
                 if not os.path.exists(path1): continue
                 sg = cv2.imread(path1)
               #  CatName = name[name.find("Class_") + 7:name.find("__ClasID__")]

                 FrontMask=(((sg[:,:,0]==1)+(sg[:,:,0]==2))>0)
                 BackMask = (sg[:, :, 0] > 2)


                 #show(sg*60)
                 # OcMask = FrontMask * (sg[:, :, 1] >2)
                 #
                 # FrontMask[OcMask]=False
                 # BackMask[OcMask] = True
#=========================================================================================
                 if BackMask.sum()/FrontMask.sum()>0.05:
                     I1 = Im.copy()
                     I2 = Im.copy()
                     I1[:, :, 0] *= 1 - (FrontMask).astype(np.uint8)
                     I1[:, :, 1] *= 1 - FrontMask.astype(np.uint8)
                     I2[:, :, 0] *= 1 - BackMask.astype(np.uint8)
                     I2[:, :, 1] *= 1 - BackMask.astype(np.uint8)
                     cv2.imshow(name+"  approve/reject", cv2.resize(np.concatenate([I1, I2], axis=1), (1300, 600)))
                     while(True):
                         ch=chr(cv2.waitKey())
                         if ch=='a' or ch=='r':break
                     if ch=='r':
                         sg = cv2.imread(path1)
                         show(sg * 60)
                         sg[:,:,0][BackMask]=1
                         show(sg * 60)
                         cv2.imwrite(path1,sg)
                         #BackMask*=0
                         sg = cv2.imread(path1)
                         show(sg*60)
                         #  CatName = name[name.find("Class_") + 7:name.find("__ClasID__")]
                         FrontMask = (((sg[:, :, 0] == 1) + (sg[:, :, 0] == 2)) > 0)
                         BackMask = (sg[:, :, 0] > 2)
                     pig=True


#==============================================================================================
                 CatID = name[name.find("CatID_") + 6:name.find(".png")]
                 Pl=0
                 while (Pl>-1):
                     Pl=CatID.find("_")
                     if Pl==-1:
                         lb = int(CatID)
                     else:
                         lb=int(CatID[:Pl])

                     CatID=CatID[Pl+1:]
                     print(name)
                     print(CatID)
                     SegMaps[lb,:,:,0][FrontMask] = 1
                     SegMaps[lb,:,:,1][BackMask] = 1
                     SegMaps[lb,:,:,2][(BackMask+FrontMask)>0] = 1
                     IgnoreIf[FrontMask]=0

                     if lb in LiquidLabels:
                         C = CatLiquid
                         SegMaps[C, :, :, 0][FrontMask] = 1
                         SegMaps[C, :, :, 1][BackMask] = 1
                         SegMaps[C, :, :, 2][(BackMask + FrontMask) > 0] = 1
                     if lb in SolidLabels:
                         C=CatSolid
                         SegMaps[C,:,:,0][FrontMask] = 1
                         SegMaps[C,:,:,1][BackMask] = 1
                         SegMaps[C,:,:,2][(BackMask + FrontMask) > 0] = 1
                     if lb in FilledLabels:
                         C=CatFilled
                         SegMaps[C, :, :, 0][FrontMask] = 1
                         SegMaps[C, :, :, 1][BackMask] = 1
                         SegMaps[C, :, :, 2][(BackMask + FrontMask) > 0] = 1
                     if lb in PartsLabels:
                         C = CatVParts
                         SegMaps[C, :, :, 0][FrontMask] = 1
                         SegMaps[C, :, :, 1][BackMask] = 1
                         SegMaps[C, :, :, 2][(BackMask + FrontMask) > 0] = 1
         SegMaps[5,:,:,0][IgnoreIf] = 1
#------------------------------------Save-----------------------------------------------------------------------------------
         if not os.path.exists(SemDir): os.mkdir(SemDir)
         for i in range(1,SegMaps.shape[0]):
             if SegMaps[i].sum()==0: continue
             name=SemDir+"//"+str(i)+"_"+Cat[i]+".png"
             cv2.imwrite(name,SegMaps[i])
#------------------------------------display--------------------------------------------------------------------------------
         if pig==True:
             for i in range(1,17):
                 name=SemDir+"//"+str(i)+"_"+Cat[i]+".png"
                 if not os.path.exists(name): continue
                 Im = cv2.imread(DirName + "/Image.png")
                 Sg=cv2.imread(name)

                 I1=Im.copy()
                 I2=Im.copy()
                 if Sg.sum()==0: continue
                 I1[:, :, 0] *= 1 - Sg[:, :, 0]
                 I1[:, :, 1] *= 1 - Sg[:, :, 0]
                 I2[:, :, 0] *= 1 - Sg[:, :, 1]
                 I2[:, :, 1] *= 1 - Sg[:, :, 1]
                 cv2.imshow(name,cv2.resize(np.concatenate([I1,I2],axis=1),(1300,600)))
                 cv2.waitKey()
                 cv2.destroyAllWindows()






###########################################################################################################################
InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
SubDir=[r"Material",r"Parts",r"Vessel"]



GenerateSemanticMap(InDir,SubDir)