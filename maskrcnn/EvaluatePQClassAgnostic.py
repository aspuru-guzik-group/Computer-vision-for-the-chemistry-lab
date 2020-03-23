import numpy as np
import cv2
import os


####################INPUT SIMPLE ANNOTATION GROUND TRUTH AND PREDICTED to get PQ evaluation#####################################################################
GTDir = "../ChemLabScapeDataset/Simple/Train/Instance"
PredDir = "coco_mixed/trainAnno0.3"

########################################Stattics collection structures############################################################################################################



tp = np.zeros([3])
fp = np.zeros([3])
fn = np.zeros([3])
liou = [[],[],[]]

#------------go over all files-----------------------------------------------------------------------
for fname in os.listdir(GTDir):
     GTann=cv2.imread(GTDir+"/"+fname)
     if not os.path.isfile(PredDir + "/" + fname):
         print(fname)
         continue
     Predann = cv2.imread(PredDir + "/"+ fname)
#-------------------------------run evaluation---------------------------------------------------------------------------------------------------------------

     for u in range(3):
         Gan=GTann[:,:,u]
         Pan=Predann[:,:,u]
         Ignore=(Gan==254)
         Pan[Ignore] = 0
         Gan[Ignore] = 0
         p=0
         for i1 in range(1,Gan.max()+1):
             for i2 in range(1, Pan.max() + 1):
                 Inter=((Pan==i2)*(Gan==i1)).sum()
                 Union=(Pan==i2).sum()+(Gan==i1).sum()-Inter+0.00001
                 IOU=Inter/Union
                 if IOU>0.5:
                     liou[u].append(IOU)
                     tp[u] += 1
                     p+=1 # count false positive per image
                     break

         fn[u]+=Gan.max()-p
         fp[u]+=Pan.max()-p
#------------------------------------------------------------------------------------------------------------------------------------------
print("=====================================================================================================================================")
print("=====================================================================================================================================")
print("=====================================================================================================================================")
RQ=np.zeros([3])
SQ=np.zeros([3])
PQ=np.zeros([3])
SuperCat=["Material","Part","Vessel"]
for u in range(3):
     RQ[u]=tp[u]/(tp[u]+fn[u]*0.5+fp[u]*0.5)
     SQ[u]=np.mean(liou[u])
     PQ[u]=RQ[u]*SQ[u]
     print(SuperCat[u]+" PQ="+str(PQ[u])+" RQ="+str(RQ[u])+" SQ="+str(SQ[u]))

