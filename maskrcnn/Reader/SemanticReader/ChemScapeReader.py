import cv2
import numpy as np
import random
import CategoryDictionary as CatDic

#############################################################################################
def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
##############################################################################################
CatName={}
CatName[1]='Vessel'
CatName[2]='V_Label'
CatName[3]='V_Cork'
CatName[4]='V_Parts_GENERAL'
CatName[5]='Ignore'
CatName[6]='Liquid_GENERAL'
CatName[7]='Liquid Suspension'
CatName[8]='Foam'
CatName[9]='Gel'
CatName[10]='Solid_GENERAL'
CatName[11]='Granular'
CatName[12]='Powder'
CatName[13]='Solid Bulk'
CatName[14]='Vapor'
CatName[15]='Other Material'
CatName[16]='Filled'
###############################################################################################


#Reader for the coco panoptic data set for pointer based image segmentation
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
import random
############################################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r"\ChemLabScapeDataset_Finished\Annotations\\", MaxBatchSize=100,MinSize=250,MaxSize=1000,MaxPixels=800*800*5,TrainingMode=True):

        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.ClassBalance=False
# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.AnnList = [] # Image/annotation list
        self.AnnByCat = {} # Image/annotation list by class

        for i in CatName:
           self.AnnByCat[CatName[i]]=[]
        uu=0

        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(MainDir):
            SemDir=MainDir+"/"+AnnDir+r"//Semantic//"
            if not os.path.isdir(SemDir): continue
            self.AnnList.append(MainDir+"/"+AnnDir)
            for Name in os.listdir(SemDir):
                i=int(Name[:Name.find("_")])
                self.AnnByCat[CatName[i]].append(MainDir+"/"+AnnDir)
            uu+=1
        if TrainingMode:
            for i in self.AnnByCat: # suffle
                    np.random.shuffle(self.AnnByCat[i])
            np.random.shuffle(self.AnnList)

        self.CatNum={}
        for i in self.AnnByCat:
              print(str(i)+") Num Examples="+str(len(self.AnnByCat[i])))
              self.CatNum[i]=len(self.AnnByCat[i])
        print("Total=" + str(len(self.AnnList)))
        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize(self,Img, AnnMap,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================

        h,w,d=Img.shape
        Bs = np.min((h/Hb,w/Wb))
        if Bs<1 or Bs>1.5:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(h / Bs)+1
            w = int(w / Bs)+1
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for i in CatName:
                AnnMap[CatName[i]] = cv2.resize(AnnMap[CatName[i]], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
 # =======================Crop image to fit batch size===================================================================================


        if np.random.rand()<0.6:
            if w>Wb:
                X0 = np.random.randint(w-Wb)
            else:
                X0 = 0
            if h>Hb:
                Y0 = np.random.randint(h-Hb)
            else:
                Y0 = 0

            Img=Img[Y0:Y0+Hb,X0:X0+Wb,:]
            for i in CatName:
                AnnMap[CatName[i]] = AnnMap[CatName[i]][Y0:Y0+Hb,X0:X0+Wb,:]

        if not (Img.shape[0]==Hb and Img.shape[1]==Wb):
            Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
            for i in CatName:
                AnnMap[CatName[i]] = cv2.resize(AnnMap[CatName[i]], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,AnnMap
        # misc.imshow(Img)
######################################################Augmented Image##################################################################################################################################
    def Augment(self,Img,AnnMap,prob):
        Img=Img.astype(np.float)
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            for i in CatName:
               AnnMap[CatName[i]] = np.fliplr(AnnMap[CatName[i]])
        if np.random.rand()<0.5:
            Img = Img[..., :: -1]


        if np.random.rand() < prob: # resize
            r=r2=(0.3 + np.random.rand() * 1.7)
            if np.random.rand() < prob*2:
                r2=(0.5 + np.random.rand())
            h = int(Img.shape[0] * r)
            w = int(Img.shape[1] * r2)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for i in CatName:
                AnnMap[CatName[i]] =  cv2.resize(AnnMap[CatName[i]], dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        # if np.random.rand() < prob/3: # Add noise
        #     noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9
        #     Img *=noise
        #     Img[Img>255]=255
        #
        # if np.random.rand() < prob/3: # Gaussian blur
        #     Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < prob*2:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img>255]=255

        if np.random.rand() < prob:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,AnnMap
########################################################################################################################################################
# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, pos, Hb=-1, Wb=-1):
# -----------------------------------Image and resize-----------------------------------------------------------------------------------------------------
            if self.ClassBalance: # pick with equal class probability
                while (True):
                     CL=random.choice(self.AnnByCat.keys())
                     CatSize=len(self.AnnByCat[CL])
                     if CatSize>0: break

                Nim = np.random.randint(CatSize)
               # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
                AnnDir=self.AnnByCat[CL][Nim]
            else: # Pick with equal class probabiliry
                Nim = np.random.randint(len(self.AnnList))
                AnnDir=self.AnnList[Nim]
                CatSize=len(self.AnnList)

            Img = cv2.imread(AnnDir+"/"+"Image.png")  # Load Image
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
#-------------------------Read annotation--------------------------------------------------------------------------------
            AnnDir+="/Semantic/"
            AnnMasks={}
            for i in CatName:
                path=AnnDir+"/"+str(i)+"_"+CatName[i]+".png"
                if os.path.exists(path):
                    AnnMasks[CatName[i]]=cv2.imread(path)  # Load mask
                else:
                    AnnMasks[CatName[i]] = np.zeros(Img.shape)  # Load mask
#-------------------------Augment-----------------------------------------------------------------------------------------------
            Img,AnnMap=self.Augment(Img,AnnMasks,np.min([float(1000/CatSize)*0.5+0.06+1,1]))
#-----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            if not Hb==-1:
               Img, AnnMap = self.CropResize(Img, AnnMap, Hb, Wb)

#---------------------------------------------------------------------------------------------------------------------------------
            self.BImg[pos] = Img
            for i in CatName:

                CN=CatName[i]
                if CN == 'Ignore':
                   self.BIgnore[pos] = AnnMap[CN][:, :, 0]
                else:
                    self.BAnnMapsFR[CN][pos] = AnnMap[CN][:, :, 0]
                    self.BAnnMapsBG[CN][pos] = AnnMap[CN][:, :, 1]


############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))


        
        #====================Start reading data multithreaded===========================================================
        self.BAnnMapsFR={}
        self.BAnnMapsBG = {}
        self.thread_list = []
        self.BIgnore = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BImg = np.zeros([BatchSize, Hb, Wb,3], dtype=float)
        for i in CatName:
              CN=CatName[i]
              if CN=='Ignore': continue
              self.BAnnMapsFR[CN] = np.zeros([BatchSize, Hb, Wb], dtype=float)
              self.BAnnMapsBG[CN] = np.zeros([BatchSize, Hb, Wb], dtype=float)
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="thread"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
        self.itr+=BatchSize
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
    def LoadBatch(self):
# Load batch for training (muti threaded  run in parallel with the training proccess)
# return previously  loaded batch and start loading new batch
            self.WaitLoadBatch()
            Imgs=self.BImg
            Ignore=self.BIgnore
            AnnMapsFR = self.BAnnMapsFR
            AnnMapsBG = self.BAnnMapsBG
            self.StartLoadBatch()
            return Imgs, Ignore, AnnMapsFR, AnnMapsBG

############################Load single data with no augmentation############################################################################################################################################################
    def LoadSingle(self):
       # print(self.itr)
        if self.itr>=len(self.AnnList):
            self.epoch+=1
            self.itr=0
        AnnDir = self.AnnList[self.itr]
        self.itr+=1
        Img = cv2.imread(AnnDir + "/" + "Image.png")  # Load Image
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
        # -------------------------Read annotation--------------------------------------------------------------------------------
        AnnDir += "/Semantic/"
        AnnMasks = {}
        for i in CatName:
            path = AnnDir + "/" + str(i) + "_" + CatName[i] + ".png"
            if os.path.exists(path):
                AnnMasks[CatName[i]] = cv2.imread(path)  # Load mask
            else:
                AnnMasks[CatName[i]] = np.zeros(Img.shape)  # Load mask

        Ignore=AnnMasks['Ignore'][:,:,0]
        del AnnMasks['Ignore']
        return Img, AnnMasks,Ignore, self.itr>=len(self.AnnList)
