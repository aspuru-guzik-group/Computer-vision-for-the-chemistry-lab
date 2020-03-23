import cv2
import numpy as np
import matplotlib.pyplot as plt
import ChemScapeReader as SemanticReader
Reader=SemanticReader.Reader(MainDir=r"../../../ChemLabScapeDataset/TrainAnnotations//",MaxBatchSize=6,MinSize=250,MaxSize=1000,MaxPixels=800*800*5,TrainingMode=True)
while (True):
    Imgs, Ignore, AnnMapsFR, AnnMapsBG=Reader.LoadBatch()


    for i in range(Imgs.shape[0]):
        Im = Imgs[i].copy()
        Im[:, :, 1] *= 1- Ignore[i]
        for an in AnnMapsFR:
            I=Im.copy()
            if AnnMapsFR[an][i].sum()>0:
                I[:, :, 0] *= 1 - AnnMapsFR[an][i]
                I[:, :, 2] *= 1 - AnnMapsBG[an][i]
                plt.imshow(I.astype(int))
                plt.title(an)
                plt.show()


