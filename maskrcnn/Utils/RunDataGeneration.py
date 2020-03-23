#Generate training data for vessel semantic segmentation, using related class from the COCO panoptic dataset
#http://cocodataset.org/#download # Download Train images and train annotation for the pnoptic set and set path in the input dirs variables


import CocoPanopticToSemanticMap as Generator
############################################Input and ouput dir location################################################################################################################################################
ImageDir="../../ChemLabScapeDataset/train2017" # image folder (coco training) train set
AnnotationDir="../../ChemLabScapeDataset/panoptic_train2017" # annotation maps folder from coco panoptic train set
DataFile="../../ChemLabScapeDataset/panoptic_train2017.json" # Json Data file coco panoptic train set


OutDir="../../ChemLabScapeDataset/COCO_2017_related_complex" # Output Dir
###############################Vessel cats in COCO###################################################################################################################


VesselCats = [44,46,47,51,86] # Cats to use as vessel #[44] 'bottle',[46] 'wine glass',[47] = 'cup',[51] = 'bowl', [86] = 'vase'
IgnoreCats = [70,50,64,196] # Unclear wether they are vessel or not.  To ignore #70 toilet 81 sink 50 spoon, 196,food-other-merged 64,potted plant
##########################################################################################################################################################
x=Generator.Generator(ImageDir,AnnotationDir,OutDir, DataFile, VesselCats=VesselCats,IgnoreCats=IgnoreCats) # Create class
x.Generate() # Run conversion