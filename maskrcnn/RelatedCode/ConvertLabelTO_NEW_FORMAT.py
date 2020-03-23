# Convert from original format to new format
import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc

###############################################################################################



def ConvertToNewFormat(ImFolder,InstDir,SegDir,OutFolder,SubDir):
    if not os.path.exists(OutFolder): os.mkdir(OutFolder)

    #######################Get Category###########################################################################
    CategoryFile=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\MergedSetAll\Annotations/ClassList.txt"
    CatFile=open(CategoryFile)
    Lines=CatFile.readlines()
    CatsDic={}
    for L in Lines:
        CatsDic[int(L[:L.find("\t")])]=L[L.find("\t")+1:L.find("\n")]
    #########################Get List of files############################################################################
    for fi,ImName in enumerate(os.listdir(ImFolder)):
         #if not (".jpg"): continue
         print(ImName)
         Folder=OutFolder+ImName[:-4]+"//"
         if not os.path.exists(Folder): os.mkdir(Folder)

         Img=cv2.imread(ImFolder+ImName)
         cv2.imwrite(Folder+"/Image.png",Img)
         Folder=Folder+"/"+SubDir+"/"
         if not os.path.exists(Folder): os.mkdir(Folder)

         Inst = cv2.imread(InstDir + "/" + ImName[:-4]+".png")
         Sem = cv2.imread(SegDir + "/" + ImName[:-4]+".png")

    #-----------------------Write Segment----------------------------------------------------------------------------
         k=0
         for i in range(1,np.max(Inst)+1):
             Seg1=np.max(Inst==i,axis=2)
             if Seg1.sum() == 0:
                 print()
                 continue
             Label1=int(np.round((Sem*(Inst == i)).sum()/ (Seg1.sum())))
             LabelName1 = CatsDic[Label1]
             k+=1
             path=Folder+"/"+str(k)+"__Class__"+LabelName1+"__ClasID__"+str(Label1)+".png"
             cv2.imwrite(path,Seg1.astype(np.uint8))
#######################################################################################################################3333

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
ImFolder = r"C:\Users\Sagi\Desktop\NewChemistryDataSet\MergedSetAll\Images\\"  # input images
OutFolder = r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\\"


InstDir = r"C:\Users\Sagi\Desktop\NewChemistryDataSet\MergedSetAll\Annotations\PartsInstance\\"
SegDir = r"C:\Users\Sagi\Desktop\NewChemistryDataSet\MergedSetAll\Annotations\PartsSemantic\\"
ConvertToNewFormat(ImFolder, InstDir, SegDir, OutFolder, "Parts")
