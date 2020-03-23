# Evaluate PQ RQ and SQ panotic statitics standart way (include classification errors)

import json
import numpy as np
import cv2
from tabulate import tabulate
import os


MinSegmentSizePixels=0 # Minoum number of pixels in instance to be evaluate
####################INPUT  ANNOTATION GROUND TRUTH AND PREDICTED folders in simple format#####################################################################
GTDir = "../ChemLabScapeDataset/Simple/Train/Instance"
GTDataFile="../ChemLabScapeDataset/Simple/Train/InstCategory.json" # GT simple json file

PredDir = "coco_mixed/trainAnno0.6"
PredDataFile="coco_mixed/trainAnno0.6/train.json" #Predicted data file (simple format)

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



MaterialCats={'Liquid GENERAL','Liquid Suspension','Foam','Gel','Solid GENERAL','Granular','Powder','Solid Bulk','Vapor','Other Material'}
PartsCats={'V Label','V Cork','V Parts GENERAL'}
################################################################################################################################################################################################################3
OrderCats=['Liquid GENERAL','Solid GENERAL','Liquid Suspension','Foam','Powder','Granular','Solid Bulk','Vapor','Gel','V Label','V Cork','V Parts GENERAL']
##################################load json class data######################################################################################################################################################################

with open(PredDataFile) as json_file:
    PredDic = json.load(json_file,parse_int=int)


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
for cn in CatName: # per class statitics
    dictp[CatName[cn]] = 0
    dicfp[CatName[cn]] = 0
    dicfn[CatName[cn]] = 0
    dicliou[CatName[cn]] = []

    fp_cat[CatName[cn]]=0
    fn_cat[CatName[cn]]=0
    tp_cat[CatName[cn]]=0


##################################################run evaluation#################################################################
for fname in os.listdir(GTDir): # scan all files

     GTann=cv2.imread(GTDir+"/"+fname) #GT annotation
     if not os.path.isfile(PredDir + "/" + fname):
         print(fname)
         continue
     Predann = cv2.imread(PredDir + "/" + fname) # pred annotation

#-----------------------------calculate pq per channel statitics (materials/parts/vessels)_-----------------------------------------------------------------------------------------------------------------

     for u in range(3):
         Gan=GTann[:,:,u]
         Pan=Predann[:,:,u]
         Ignore=(Gan==254)
         Pan[Ignore] = 0
         Gan[Ignore] = 0
         p=0


#-----------------Count true positive and false negative and iou (sq)-----------------------------------------------------------------------------------------
         for i1 in range(1,Gan.max()+1): # Go over alll gt segment and try to find matching predictd segment
             if (Gan == i1).sum() < MinSegmentSizePixels: continue
             gtCats={}
             if u == 0:
                 gtCats = GTDic[fname[:-4]]['MaterialCats'][str(i1)] # Get segment GT catgories

                 IsMultiphase=0
                 if i1 in GTDic[fname[:-4]]['MultiPhaseMaterial']:
                     IsMultiphase=1
                 MultiPhaseGt[IsMultiphase]+=len(gtCats) # Count number of materials that are part of multiphase system

             if u == 1:  gtCats = GTDic[fname[:-4]]['PartCats'][str(i1)] # Get GT segment catgories
             IOU = 0

             for i2 in range(1, Pan.max() + 1): # Go over prediction  classed
                 pdCats = {}
                 if u == 0: pdCats = PredDic[fname[:-4]]['MaterialCats'][str(i2)] # Get predicted cats
                 if u == 1: pdCats = PredDic[fname[:-4]]['PartCats'][str(i2)]# Get predicted cats
                 Inter=((Pan==i2)*(Gan==i1)).sum()
                 Union=(Pan==i2).sum()+(Gan==i1).sum()-Inter+0.00001
                 IOU=Inter/Union
                 if IOU>0.5:


                   #  p+=1 # count false positive per image
                     if u==2: break
                     for m in gtCats: # If to segments overlap check if predicted and ground truth segments matcg
                         if m in  pdCats:
                             if u == 0:
                                 MultiPhaseDet[IsMultiphase] += 1
                                 MultiPhaseIOU[IsMultiphase].append(IOU)
                             tp_cat[m]+=1
                             liou[u].append(IOU)
                             tp[u] += 1
                             dicliou[m].append(IOU)
                             dictp[m] += 1
                         else:
                             fn_cat[m] += 1
                             dicfn[m] += 1
                             fn[u] += 1
                     break
             if IOU<0.5:
                fn[u]+=len(gtCats)
                for m in gtCats:
                    dicfn[m]+=1
#-----------------Count false postive-----------------------------------------------------------------------------------------
         for i2 in range(1,Pan.max()+1):
             if (Pan == i2).sum() < MinSegmentSizePixels: continue
             pdCats={}
             IOU = 0
             if u == 0: pdCats = PredDic[fname[:-4]]['MaterialCats'][str(i2)] # Get predcition  segment classes
             if u == 1: pdCats = PredDic[fname[:-4]]['PartCats'][str(i2)]
             for i1 in range(1, Gan.max() + 1):
                 gtCats={}
                 if u == 0: gtCats = GTDic[fname[:-4]]['MaterialCats'][str(i1)] # Get ground truth segment classes
                 if u == 1: gtCats = GTDic[fname[:-4]]['PartCats'][str(i1)]
                 Inter=((Pan==i2)*(Gan==i1)).sum()
                 Union=(Pan==i2).sum()+(Gan==i1).sum()-Inter+0.00001
                 IOU=Inter/Union
                 if IOU>0.5:
                     for m in pdCats:
                         if m not in gtCats: # If segment classes dont match add the prediction classes to false positive (note that the truth positves we handled in previouoop)
                             fp_cat[m] += 1
                             fp[u] += 1
                             dicfp[m] += 1

                     break
             if IOU<0.5:
                fp[u]+=len(pdCats)
                for m in pdCats:
                    dicfp[m]+=1
#----------------------------------------------Display statistics--------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------
print("#####################################statics WithClasss#################################################################################################################")
print("Input folder "+PredDir)
print("Min number pixels in segment="+str(MinSegmentSizePixels))
print("==========================================General mean for all instances===========================================================================================")
SuperCat=["Material","Part","Vessel"]
head = ["class", "PQ", "RQ", "SQ", "Recall", "Num Instace"]
content = []
for u in range(3):
     RQ=tp[u]/(tp[u]+fn[u]*0.5+fp[u]*0.5)
     SQ=np.mean(liou[u])
     PQ=RQ*SQ
     Recall = tp[u] / (tp[u] + fn[u]+0.00001)
     numsamples = tp[u] + fn[u]
     content.append([SuperCat[u], PQ, RQ, SQ, Recall, numsamples])
table = tabulate(content, head, tablefmt="fancy_grid")
print(table)

print("==========================================Per Class===========================================================================================")
mPQ=0
mRQ=0
mSQ=0
ncl=0
mRecall=0
TotalFPMater=0
head = ["class", "PQ", "RQ", "SQ", "Recall", "Num Instace"]
content = []
for  u in OrderCats:#dictp:
     if (u not in PartsCats)  and (u not in MaterialCats): continue
     RQ=dictp[u]/(dictp[u]+dicfn[u]*0.5+dicfp[u]*0.5+0.000001)
     SQ=np.mean(dicliou[u])
     PQ=RQ*SQ
     Recall = dictp[u] / (dictp[u] + dicfn[u])
     content.append([u, PQ, RQ, SQ, Recall, dictp[u] + dicfn[u]])
     if np.isnan(PQ) or np.isnan(RQ) or np.isnan(SQ) or (u not in MaterialCats): continue
     mPQ += PQ
     mRQ += RQ
     mSQ += SQ
     mRecall += Recall
     TotalFPMater+=dicfp[u]
     ncl += 1
table = tabulate(content, head, tablefmt="fancy_grid")
print(table)
#print("==========================================General mean for all classes===========================================================================================")
print("Mean For all material classes\t"+str(mPQ/ncl)+"\t"+str(mRQ/ncl)+"\t"+str(mSQ/ncl)+"\t"+str(mRecall/ncl))


print("==========================================Accuracy for one phase and multphase system===========================================================================================")
head = ["class", "PQ", "RQ", "SQ", "Recall", "Num Instace"]
content = []
for i in range(2):
    TotalGTMater=np.sum(MultiPhaseGt)
    TP = MultiPhaseDet[i]
    FN = MultiPhaseGt[i]-MultiPhaseDet[i]
    FP = TotalFPMater * MultiPhaseGt[i] / TotalGTMater

    RQ = TP / (TP + FP * 0.5 + FN * 0.5 + 0.000001)
    SQ = np.mean(MultiPhaseIOU[i])
    PQ = RQ * SQ
    Recall = TP / (TP + FN)
    content.append([i+1, PQ, RQ, SQ, Recall, MultiPhaseGt[i]])

table = tabulate(content, head, tablefmt="fancy_grid")
print(table)
print("###################################################################################################################################################################################")


print("==========================================Classification accuracy===========================================================================================")
print("Class\tRecall\tPrecision\tNum GtExamples")
head = ["class", "Recall", "Precision", "Num GtExamples"]
content = []
for m in OrderCats:#tp_cat:
    if tp_cat[m]+fp_cat[m]+fn_cat[m]>0:
        content.append([m, tp_cat[m]/(0.01+tp_cat[m]+fn_cat[m]), tp_cat[m]/(0.01+tp_cat[m]+fp_cat[m]), tp_cat[m]+fn_cat[m]])

table = tabulate(content, head, tablefmt="fancy_grid")
print(table)
print("###################################################################################################################################################################################")