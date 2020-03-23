import numpy
import json
import cv2
import numpy as np
import os
import scipy.misc as misc
def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
###############################################################################################
def FindIntersection(InDir,MatDir, VesselDir):
    pp=0
    for DirName in os.listdir(InDir):
         pp+=1
         print(pp)
         DirName=InDir+"/"+DirName
         MSgDir = DirName + "/" + MatDir + "//"
         VSgDir = DirName + "/" + VesselDir + "//"
         if not os.path.isdir(MSgDir):
                      print(MSgDir)
                      continue

         # listfile=[]
         # for fl in os.listdir(MSgDir):
         #     if ".png" in fl:
         #         listfile.append(fl)

         # l=len(listfile)
         k=0
         Im = cv2.imread(DirName+"/Image.png")

         #for  i in range(l):

         for mfile in os.listdir(MSgDir):
             NVessels = 0
             path1=MSgDir+"/"+mfile
             if not os.path.exists(path1):continue
             msg = cv2.imread(path1,0)
             if msg.sum()==0:
                 os.remove(path1)
                 print(path1+"File Removed!")
                 continue

             # CatName=listfile[i][listfile[i].find("Class__")+7:listfile[i].find("__ClasID__")]
             # CatID=listfile[i][listfile[i].find("ClasID__")+8:listfile[i].find(".png")]
             emsg=np.expand_dims(msg,axis=2)
             for vfile in os.listdir(VSgDir):
                 path2 = VSgDir + "/" + vfile
                 if not os.path.exists(path2): continue
                 vsg = cv2.imread(path2, 0)
                 inter=((vsg*msg)>0)#.astype(np.uint8)
                 print(path1)
                 print(path2)
                 if (inter).sum()/((msg>0).sum())<0.8:
                     if (inter).sum()/((msg>0).sum())>0.01:
                         #..........................................
                         Txt="  i(in vessel) f(front of vessel) a(after vessel)"
                         Im1=Im.copy()
                         Im1[:,:,0] *= 1-vsg
                         Im1[:, :, 2] *= 1 - msg

                         cv2.imshow(Txt+"2", cv2.resize(Im1,(500,500)))
                         cv2.imshow(Txt, cv2.resize(np.concatenate([vsg, msg], axis=1) * 250,(1000,500)))

                         while (True):
                             ch = chr(cv2.waitKey())
                             if ch=='i' or ch=='f' or ch=='a': break

                         cv2.destroyAllWindows()
                         if ch=='i':
                             emsg = np.concatenate([emsg, np.expand_dims(vsg, axis=2)], axis=2)
                             NVessels+=1
                         if ch=='a':
                             msg[inter > 0]=5
                             emsg[:,:,0]=msg
                 else:
                     emsg = np.concatenate([emsg, np.expand_dims(vsg,axis=2)],axis=2)
                     NVessels += 1
                 if NVessels>2:
                     print("error")
                     print(path1)
                     print(path2)
                     show(Im)
                     show(msg*50)

                     exit(0)

             if emsg.shape[2]==2:
                 emsg = np.concatenate([emsg, np.expand_dims(vsg*0,axis=2)],axis=2)
             cv2.imwrite(path1, emsg)



         ###############################################################################################################
             # sg = cv2.imread(path1)
             # Im = cv2.imread(DirName + "/Image.png")
             # cv2.imshow("results", sg*50)
             # Im[:,:,0]*=1-(sg[:,:,0]>0).astype(np.uint8)
             # Im[:, :, 1] *= 1 - (sg[:, :, 1] > 0).astype(np.uint8)
             # cv2.imshow("im",Im)
             #
             #
             # for i in range(sg.shape[2]):
             #   print("------------------------------------------------------------------------")
             #   print(str(i))
             #   print(np.unique(sg[:,:,i]))
             #   cv2.imshow(str(i) +"  ", sg[:,:,i] * 35)
             #
             # cv2.waitKey()
             # cv2.destroyAllWindows()
         ###########################################################################################################################
         os.rename(MSgDir,MSgDir.replace(MatDir,MatDir+"V"))

InDir=r"C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Temp\\"##C:\Users\Sagi\Desktop\NewChemistryDataSet\NewFormat\Instance\\"
MatDir=r"PartsVi"
VesselDir=r"VesselV"
# FindIntersection(InDir, SubDir)

FindIntersection(InDir,MatDir, VesselDir)