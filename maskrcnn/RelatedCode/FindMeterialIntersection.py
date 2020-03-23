import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc
# Find material segments with high intersection and assign priority
###############################################################################################
def FindIntersection(InDir,SubDir):
    pp=0
    for DirName in os.listdir(InDir):
         pp+=1
         print(pp)
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
         Im = cv2.imread(DirName+"/Image.png")

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
                 inter=((sg1*sg2)>0)#.astype(np.uint8)
                 CatName2 = listfile[f][listfile[f].find("Class__") + 7:listfile[f].find("__ClasID__")]
                 if (inter).sum()/(np.max([sg1.sum(),sg2.sum()]))>0.02:

                     #..........................................
                     Txt=CatName+"  [z]1<--([5]h botth) ([y] lboth) ([r] vl both)-->0[m]   "+CatName2
                     Im1=Im.copy()
                     Im1[:,:,0] *= 1-sg1
                     Im1[:, :, 2] *= 1 - sg2

                     cv2.imshow(Txt+"2", cv2.resize(Im1,(500,500)))
                     cv2.imshow(Txt, cv2.resize(np.concatenate([sg1, sg2], axis=1) * 250,(1000,500)))

                     while (True):
                         ch = chr(cv2.waitKey())
                         if ch=='1' or ch=='0' or ch=='z' or ch=='m' or ch=='y' or ch=='r' or ch=='5': break

                     cv2.destroyAllWindows()
                     if ch=='1':
                         sg1[inter>0] = 2 # Priority
                         sg2[inter>0] = 3 # Low Priority
                     if ch=='0':
                         sg1[inter>0] = 3 # Low Priority
                         sg2[inter>0] = 2 # Priority

                     if ch=='z':
                         sg1[inter>0] = 2 # Priority
                         sg2[inter>0] = 4 #  Very Low Priority
                     if ch=='m':
                         sg1[inter>0] = 4 # Very Low Priority
                         sg2[inter>0] = 2 # Priority
                     if ch=='y':
                         sg1[inter>0] = 3 #  Low Priority
                         sg2[inter>0] = 3 # Low Priority
                     if ch == 'r':
                         sg1[inter > 0] = 4  # Very Low Priority
                         sg2[inter > 0] = 4  # Very Low Priority
                     if ch == '5':
                         sg1[inter > 0] = 2  #  Priority
                         sg2[inter > 0] = 2  #  Priority


                     cv2.imwrite(path1,sg1)
                     cv2.imwrite(path2, sg2)
         ###############################################################################################################
                     # sg1 = cv2.imread(path1, 0)
                     # sg2 = cv2.imread(path2, 0)
                     # cv2.imshow("results", Im1)
                     # cv2.imshow("res", np.concatenate([sg1, sg2], axis=1) * 60)
                     # cv2.waitKey()
                     # cv2.destroyAllWindows()
         ###########################################################################################################################
         os.rename(SgDir,SgDir.replace(SubDir,SubDir+"V"))

InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
SubDir=r"Parts"
FindIntersection(InDir, SubDir)