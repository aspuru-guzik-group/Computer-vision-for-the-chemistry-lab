import os
import cv2
import numpy as np
import shutil
###########################Display image##################################################################
def show(Im,Name="img"):
    cv2.imshow(Name,Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
##################################################################################################################################################################
#Split binary mask correspond to a singele segment into connected components
def GetConnectedSegment(Seg):
        [NumCCmp, CCmpMask, CCompBB, CCmpCntr] = cv2.connectedComponentsWithStats(Seg.astype(np.uint8))  # apply connected component
        Mask=np.zeros([NumCCmp,Seg.shape[0],Seg.shape[1]],dtype=bool)
        BBox=np.zeros([NumCCmp,4])
        Sz=np.zeros([NumCCmp],np.uint32)
        for i in range(1,NumCCmp):
            Mask[i-1] = (CCmpMask == i)
            BBox[i-1] = CCompBB[i][:4]
            Sz[i-1] = CCompBB[i][4] #segment Size
        return Mask,BBox,Sz,NumCCmp-1
############################################################################################################################
##############################################################################################
MainDir=r"C:\Users\Sagi\Desktop\CHEMSCAPE\ChemLabScapeDataset\TestAnnoatations\\"
for AnnDir in os.listdir(MainDir):

    VesDir = MainDir + "/" + AnnDir + r"//Vessel//"
    SemDir = MainDir + "/" + AnnDir + r"//Semantic//"
    EmptyDir = MainDir + "/" + AnnDir + r"//EmptyRegions//"
    Img = cv2.imread(MainDir +"/"+ AnnDir + "/Image.png")
    if os.path.isdir(EmptyDir): shutil.rmtree(EmptyDir)
    os.mkdir(EmptyDir)
    #___________________________________________________________________________________________________________________
    NumEmptyInst=0
    IsFilled=False
    IsVapor=False
    if os.path.exists(SemDir+"16_Filled.png"):
                Filled=cv2.imread(SemDir+"16_Filled.png")[:,:,0]>0
                IsFilled = True
    if os.path.exists(SemDir+"14_Vapor"):
                Vapor=cv2.imread(SemDir+"14_Vapor.png")[:,:,0]>0
                IsVapor = True
    for Name in os.listdir(VesDir):


                path=VesDir+Name
                if not os.path.exists(path): continue
                Ves=cv2.imread(path)

                Ves[:,:,1]*=0
                if not 7 in Ves:
                   Ves[:,:,2]*=0
                cv2.imwrite(path,Ves)
                print(path)
                # show(Ves*30)
                Ves[:, :, 1]=Ves[:, :, 0]
                Mask=Ves[:, :, 0]>0
                if IsFilled:
                    Mask[Filled]=0
                if IsVapor:
                    Mask[Vapor]=0
               # show((Img/2+Ves*100).astype(np.uint8))
                #show(Ves * 100, str(NumEmptyInst)+"ALL VESSEL")
                Mask,BBox,Sz,NumCCmp=GetConnectedSegment(Mask.astype(np.uint8))
                for i in range(NumCCmp):
                      if Mask[i].sum()<1200: continue
                      Inst=Ves.copy()
                      Inst[:, :, 0]=Inst[:, :, 1]*Mask[i]
                #     NumEmptyInst+=1
                      cv2.imwrite(EmptyDir+"//"+str(NumEmptyInst)+".png",Inst)
                     # show(Inst[:,:,0]*30,str(NumEmptyInst))




                # if 7 in Ves:
                #     print("444")
                #     show(Ves*20)
                #     show((Ves==7).astype(np.uint8)*100)






