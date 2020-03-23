# Generate training data for the net using the COCO panoptic 2017 dataset take classes related to vessels and create binary maps of all thesel classes
import numpy as np
import os
import cv2
import json
from shutil import copyfile


#######################################Convert RGB image to label #####################################################################
def rgb2id(color):  # Convert annotation map from 3 channel RGB to instance
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


#########################################################################################################################
######################################Generate vessels mask from instance segments of the COCO set by merging all instance corres###################################################################################
class Generator:
    # Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir, AnnotationDir, OutDir, DataFile, AnnotationFileType="png", ImageFileType="jpg",
                 UnlabeledTag=0, VesselCats=[44, 46, 47, 51, 86], IgnoreCats=[70, 81, 81, 50]):

        self.ImageDir = ImageDir  # Image dir
        self.AnnotationDir = AnnotationDir  # File containing image annotation
        self.AnnotationFileType = AnnotationFileType  # What is the the type (ending) of the annotation files
        self.ImageFileType = ImageFileType  # What is the the type (ending) of the image files
        self.DataFile = DataFile  # Json File that contain data on the annotation of each image
        self.UnlabeledTag = UnlabeledTag  # Value of unlabled region in the annotation map (usually 0)

        self.Outdir = OutDir
        self.SemanticDir = OutDir + "/SemanticMaps/"
        self.OutImageDir = OutDir + "/Image/"
        # self.MinSegSize = 0#400
        if not os.path.exists(OutDir): os.mkdir(OutDir)
        if not os.path.exists(self.SemanticDir): os.mkdir(self.SemanticDir)
        # if not os.path.exists(self.SemanticSpecialDir): os.mkdir(self.SemanticSpecialDir)
        #
        if not os.path.exists(self.OutImageDir): os.mkdir(self.OutImageDir)

        self.VesselCats = VesselCats  # class of vessel like object [44] 'bottle',[46] 'wine glass',[47] = 'cup',[51] = 'bowl', [86] = 'vase'
        self.IgnoreCats = IgnoreCats  # ambigious classes that should be ignored[70,81,50,64,196] # Unclear wether they vessel or not to ignore #70 toilet 81 sink 50 spoon64,196] # Unclear wether they vessel or not to ignore #70 toilet 81 sink 50 spoon, 196,food-other-merged 64,potted plant
        # ) bottle]
        #  self.PickBySize = False  # Pick instances of with probablity proportional to their sizes if false all segments are picked with equal probablity
        # ........................Read data file................................................................................................................
        with open(DataFile) as json_file:
            self.AnnData = json.load(json_file)

        # -------------------Get All files in folder--------------------------------------------------------------------------------------
        self.FileList = []
        for FileName in os.listdir(AnnotationDir):
            if AnnotationFileType in FileName:
                self.FileList.append(FileName)

    ##############################################################################################################################################
    # Get annotation data for specific image from the json file
    def GetAnnnotationData(self, AnnFileName):
        for item in self.AnnData['annotations']:  # Get Annotation Data
            if (item["file_name"] == AnnFileName):
                return (item['segments_info'])

    # ############################################################################################################################################
    # #Get information for specific catagory/Class id
    def GetCategoryData(self, ID):
        for item in self.AnnData['categories']:
            if item["id"] == ID:
                return item["name"], item["isthing"]
        return "", 0

    ######################################################################################################
    # Generate list of all  segments in the image also generate semantic map
    # Given the annotation map a json data file create list of all segments and instance with info on each segment and merge all the segment
    # --------------------------Generate list of all segments--------------------------------------------------------------------------------
    def GenerateSemanticMap(self, Ann, Ann_name):
        AnnList = self.GetAnnnotationData(Ann_name)
        h, w = Ann.shape
        annos= []
        for an in AnnList:
            SemanticMap = np.zeros([h, w], dtype=np.uint8)  # 1 vessel cat, 0 other cat
            ROIMap = np.zeros([h, w], dtype=np.uint8)  # 1 Annoted region, 0 not annotated or not clear
            ct = an["category_id"]
            if ct in self.VesselCats:
                SemanticMap[Ann == an['id']] = 1
                annos.append(SemanticMap)
            if not (ct in self.IgnoreCats):
                ROIMap[Ann == an['id']] = 1

        return annos

    ####################################################Go over all files and convert annotation to vessel mask##############################################################################################
    def Generate(self):

        # for ID in range(200): # Display all cata
        #     name,Isthing=self.GetCategoryData(ID)
        #     print(str(ID)+") "+name)
        # ErrorCount=0

        for f, Ann_name in enumerate(self.FileList):  # Get label image name
            print(str(f) + ")" + Ann_name)
            Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name)  # Load Annotation
            Ann = Ann[..., :: -1]
            self.AnnColor = Ann
            Ann = rgb2id(Ann)
            SemanticMap = self.GenerateSemanticMap(Ann, Ann_name)  # Generate list of all segments in image

            #                if os.path.exists(self.SegMapDir + "/" + Ann_name): continue
            print(str(f) + ")" + Ann_name)

            if len(SemanticMap) > 0:
                os.makedirs(self.SemanticDir + '/' + Ann_name.split(".")[0])
                os.makedirs(self.SemanticDir + '/' + Ann_name.split(".")[0] + '/' + 'Vessel')
                copyfile(self.ImageDir + "/" + Ann_name.replace(".png", ".jpg"), self.SemanticDir + '/' + Ann_name.split(".")[0] + "/" + "Image.jpg")
                # -------------------------------------------Save semanic map---------------------------------------------------------------------------------------------------
                for i in range(len(SemanticMap)):
                    cv2.imwrite(self.SemanticDir + "/" + Ann_name.split(".")[0] + '/' + 'Vessel' + '/' + f"{i}_Class_Vessel_CatID_1.png", SemanticMap[i].astype(np.uint8))
