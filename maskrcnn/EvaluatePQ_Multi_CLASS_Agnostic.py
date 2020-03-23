# Create PQ,RQ,SQ  precision and recall statitics using ground truth and predicted mask in simple form,  class agnostic the predicted segment class will not be used

import json
import numpy as np
import cv2
from tabulate import tabulate
import os
MinSegmentSizePixels=0 # Minoum number of pixels in instance to be evaluate
####################INPUT SIMPLE ANNOTATION GROUND TRUTH AND PREDICTED folders #####################################################################
GTDir = "../ChemLabScapeDataset/Simple/Test/Instance"
GTDataFile="../ChemLabScapeDataset/Simple/Test/InstCategory.json" # GT simple json file

PredDir = "coco_mixed/testAnno0.6"
PredDataFile="coco_mixed/testAnno0.6/test.json" #Predicted data file (simple format)
###########################################################################################################################################################
########################################Statitics for all instances not divided to class###########################################################################################################
########################################Statitics for all instances not divided to classs############################################################################################################
########################################Statitics for all instances not divided to class###########################################################################################################
########################################Statitics for all instances not divided to classs############################################################################################################
####################################################################################################################################################################################################
tp = np.zeros([3])
fp = np.zeros([3])
fn = np.zeros([3])
liou = [[],[],[]]

#------------go over all files-----------------------------------------------------------------------
for fname in os.listdir(GTDir):
     GTann=cv2.imread(GTDir+"/"+fname) # predicted annotation
     Predann = cv2.imread(PredDir + "/" + fname)   # Gt annotation
     if not os.path.exists(PredDir + "/" + fname):
         print("missing prediction file:"+fname)
         Predann=GTann*0
     #-------------------------------run evaluation for each channel Material/Parts/Vessel---------------------------------------------------------------------------------------------------------------

     for u in range(3):
         Gan=GTann[:,:,u]
         Pan=Predann[:,:,u]
         Ignore=(Gan==254)
         Pan[Ignore] = 0
         Gan[Ignore] = 0
         p=0
         for i1 in range(1,Gan.max()+1): # Scan all GT instances
             for i2 in range(1, Pan.max() + 1): # Scan all predicted instance
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
#----------------------------------------------Display statistic--------------------------------------------------------------------------------------------
print("******************************************CLASS AGNOSITC*********************************************************")
print("=====================================================================================================================================")
print("==============================================Statistics all instances not divided by class=======================================================================================")
SuperCat=["Material","Part","Vessel"]
head = ["class", "PQ", "RQ", "SQ", "Recall"]
content = []
for u in range(3):
     RQ=tp[u]/(tp[u]+fn[u]*0.5+fp[u]*0.5+0.001)
     SQ=np.mean(liou[u])
     PQ=RQ*SQ
     Recall = tp[u] / (tp[u] + fn[u]+0.00001)
     content.append([SuperCat[u], PQ, RQ, SQ, Recall])

table = tabulate(content, head, tablefmt="fancy_grid")
print(table)
TotalFPMater=fp[0] # Total number of fp positive for material instances
TotalGTMater=tp[0]+fn[0] # # total number of material instance
######################################################################################################################################################################################################################
#############################################Statistics divided to classes############################################################################################################################################
#############################################Statistics divided to classes############################################################################################################################################
#############################################Statistics divided to classes############################################################################################################################################
#############################################Statistics divided to classes############################################################################################################################################
######################################################################################################################################################################################################################


#######################################List tof classes#################################################################################################################################################################
CatName={}
CatName[1]='Vessel'
CatName[2]='V Label'
CatName[3]='V Cork'
CatName[4]='V Parts GENERAL'
CatName[5]='Ignore'
CatName[6]='Liquid GENERAL'
CatName[7]='Liquid Suspension'
CatName[8]='Foam'
CatName[9]='Gel'
CatName[10]='Solid GENERAL'
CatName[11]='Granular'
CatName[12]='Powder'
CatName[13]='Solid Bulk'
CatName[14]='Vapor'
CatName[15]='Other Material'
CatName[16]='Filled'
#######################################################Ordered Cats####################################################################################
OrderCats=['Liquid GENERAL','Solid GENERAL','Liquid Suspension','Foam','Powder','Granular','Solid Bulk','Vapor','Gel','V Label','V Cork','V Parts GENERAL']
MaterialCats={'Liquid GENERAL','Liquid Suspension','Foam','Gel','Solid GENERAL','Granular','Powder','Solid Bulk','Vapor','Other Material'}
PartsCats={'V Label','V Cork','V Parts GENERAL'}

##################################load json with class data######################################################################################################################################################################

with open(GTDataFile) as json_file:
    GTDic = json.load(json_file,parse_int=int)
#########################################Generate statitcs collection list######################################################
MultiPhaseGt=[0,0]
MultiPhaseDet=[0,0]
MultiPhaseIOU=[[],[]]

tp = np.zeros([3])
fp = np.zeros([3])
fn = np.zeros([3])
liou = [[],[],[]]

dictp = {}
dicfp = {}
dicfn = {}
dicliou = {}

fp_cat = {}
fn_cat = {}
tp_cat = {}
for cn in CatName:
    dictp[CatName[cn]] = 0
    dicfp[CatName[cn]] = 0
    dicfn[CatName[cn]] = 0
    dicliou[CatName[cn]] = []

    fp_cat[CatName[cn]]=0
    fn_cat[CatName[cn]]=0
    tp_cat[CatName[cn]]=0

print("##############################################Creating Statitics divided by class###############################################################################################################")

##################################################run evaluation#################################################################
for fname in os.listdir(GTDir):
     #ImgData=GTDic[fname[:-4]]

     #print(fname)
     GTann=cv2.imread(GTDir+"/"+fname)
     Predann = cv2.imread(PredDir + "/" + fname)
     if not os.path.exists(PredDir + "/" + fname):
         print("missing prediction file:" + fname)
         Predann = GTann * 0
#-----------------------------calculate pq per channel (materials/parts/vessels_-----------------------------------------------------------------------------------------------------------------

     for u in range(3):
         Gan=GTann[:,:,u]
         Pan=Predann[:,:,u]
         Ignore=(Gan==254)
         Pan[Ignore] = 0
         Gan[Ignore] = 0
         p=0


#-----------------Count true positive and false negative and iou (sq)-----------------------------------------------------------------------------------------
         for i1 in range(1,Gan.max()+1): # scan over all gt instances
             if (Gan == i1).sum() < MinSegmentSizePixels: continue
             gtCats={}
             if u == 0:
                 gtCats = GTDic[fname[:-4]]['MaterialCats'][str(i1)] # get gt instance classes
                 IsMultiphase=0
                 if i1 in GTDic[fname[:-4]]['MultiPhaseMaterial']:
                     IsMultiphase=1
                 MultiPhaseGt[IsMultiphase]+=1 # Number of multiphase GT materials

             if u == 1:  gtCats = GTDic[fname[:-4]]['PartCats'][str(i1)] # get gt instance classes
             IOU = 0

             for i2 in range(1, Pan.max() + 1): # compare to all pred instances
                 Inter=((Pan==i2)*(Gan==i1)).sum()
                 Union=(Pan==i2).sum()+(Gan==i1).sum()-Inter+0.00001
                 IOU=Inter/Union
                 if IOU>0.5:
                     if u==0:
                         MultiPhaseDet[IsMultiphase] += 1 # count according to number of phases
                         MultiPhaseIOU[IsMultiphase].append(IOU)
                     liou[u].append(IOU)
                     tp[u] += 1
                     p+=1 # count false positive per image
                     if u==2: break
                     for m in gtCats: # add to the class statics
                         dicliou[m].append(IOU)
                         dictp[m] += 1
                     break
             if IOU<0.5:
                fn[u]+=1
                for m in gtCats:# add to the class statics
                    dicfn[m]+=1
#------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------Display stattics-------------------------------------------------------------------------------------
#-----------------------------------------------------Display stattics-------------------------------------------------------------------------------------
#-----------------------------------------------------Display stattics-------------------------------------------------------------------------------------
#-----------------------------------------------------Display stattics-------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
print("#####################################classs Agonostic divided Classs#################################################################################################################")
print("Input folder "+PredDir)
print("Min number pixels in segment="+str(MinSegmentSizePixels))
print("==========================================Per Class===========================================================================================")
mPQ=0
mRQ=0
mSQ=0
mRecall=0
ncl=0
head = ["class", "PQ", "RQ", "SQ", "Recall", "Num Instace"]
content = []
for  u in OrderCats:#dictp:
     fpM=TotalFPMater * (dictp[u]+dicfn[u]) / TotalGTMater
     if (u not in PartsCats)  and (u not in MaterialCats): continue
     RQ=dictp[u]/(dictp[u]+dicfn[u]*0.5+fpM*0.5+0.000001)
     SQ=np.mean(dicliou[u])
     PQ=RQ*SQ
     Recall=dictp[u]/(dictp[u]+dicfn[u]+0.000001)
     content.append([u, PQ, RQ, SQ, Recall, dictp[u]+dicfn[u]])
     if np.isnan(PQ) or np.isnan(RQ) or np.isnan(SQ) or (u not in MaterialCats): continue
     mPQ += PQ
     mRQ += RQ
     mSQ += SQ
     mRecall+=Recall
     ncl += 1
table = tabulate(content, head, tablefmt="fancy_grid")
print(table)
#print("==========================================General mean for all classes===========================================================================================")
print("Mean For all materials classes\t"+str(mPQ/ncl)+"\t"+str(mRQ/ncl)+"\t"+str(mSQ/ncl)+"\t"+str(mRecall/ncl))


print("==========================================Accuracy for one phase and multphase system===========================================================================================")
head = ["Num Phase", "PQ", "RQ", "SQ", "Recall", "Num Instace"]
content = []
for i in range(2):
    TP = MultiPhaseDet[i]
    FN = MultiPhaseGt[i]-MultiPhaseDet[i]
    FP = TotalFPMater * MultiPhaseGt[i] / TotalGTMater

    RQ = TP / (TP + FP * 0.5 + FN * 0.5 + 0.000001)
    SQ = np.mean(MultiPhaseIOU[i])
    PQ = RQ * SQ
    Recall = TP / (TP + FN)
    content.append([u, PQ, RQ, SQ, Recall, MultiPhaseGt[i]])
table = tabulate(content, head, tablefmt="fancy_grid")
print(table)
print("###################################################################################################################################################################################")


