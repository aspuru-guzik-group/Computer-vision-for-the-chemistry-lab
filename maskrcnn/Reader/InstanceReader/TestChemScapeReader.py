import cv2
import numpy as np
import matplotlib.pyplot as plt
###########################Display image##################################################################
def show(Im,Name="img")  :
    cv2.imshow(Name,Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
##############################################################################################

import ChemScapeVesselInstanceReader as ChemScapeInstanceReader
Reader= ChemScapeInstanceReader.Reader(MainDir=r"../../../ChemLabScapeDataset/TrainAnnotations//",MaxBatchSize=6,MinSize=250,MaxSize=1000,MaxPixels=800*800*5,TrainingMode=True)
while (True):
    Imgs, Ignore, InstFR, InstBG, ROI, PointerMask=Reader.LoadBatch()
    for i in range(Imgs.shape[0]):
        Im = Imgs[i].copy()

        I1=Im.copy()
        I1[:, :, 0] *= 1 - InstFR[i]
        I1[:, :, 2] *= 1 - InstFR[i]
        I2 = Im.copy()
        I2[:, :, 0] *= 1 - InstBG[i]
        I2[:, :, 2] *= 1 - InstBG[i]

        I3 = Im.copy()
        I3[:, :, 0] *= 1 - ROI[i]
        I3[:, :, 2] *= 1 - ROI[i]

        Im[:, :, 1] *= 1 - Ignore[i]
        Im[:, :, 2] *= 1 - Ignore[i]

        plt.subplot(321)
        plt.imshow(Imgs[i].astype(int))
        plt.subplot(322)
        plt.imshow(Im.astype(int))
        plt.subplot(323)
        plt.imshow(InstFR[i])
        plt.subplot(324)
        plt.imshow(InstBG[i])
        plt.subplot(325)
        plt.imshow(ROI[i])
        plt.subplot(326)
        plt.imshow(Ignore[i])
        plt.show()
        plt.imshow(cv2.resize(np.concatenate([Im,I3,I1,I2],axis=1),(1400,400)).astype(int))
        plt.show()
       # show(cv2.resize(np.concatenate([Im,I3,I1,I2],axis=1),(1400,400)))
        #show((PointerMask[i]*90+InstFR[i]*35).astype(np.uint8))


