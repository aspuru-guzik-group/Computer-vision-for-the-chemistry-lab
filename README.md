# Computer vision for recognition segmentation and classification of materials and vessels in the chemistry lab (and other settings)

The methods to in this repository are aimed to detect segment and classify vessels, and material phases, in mostly transparent containers/vessels in the chemistry lab. All the methods here are based on convolutional neural nets (CNN). The methods were trained using the Vector-LabPics dataset.
# Semantic segmentation
Network for semantic segmentation of both vessels and materials using fully convolutional net can be found here:

[PSP net] (https://github.com/aspuru-guzik-group/Semantic-segmentation-of-materials-and-vessels-in-chemistry-lab-using-FCN). 

This basically assigns one or more class per pixel. The classes includes vessel or various of materials class such as liquids,solids powder,foam... (Figure 1).

# Instance segmentation.
Nets that split the image to instances of materials and vessels (Figure 2) can be found here:


[Generator evaluator selector net (Hierarchical)] (https://github.com/aspuru-guzik-group/Instance-segmentation-of-images-of-materials-in-transparent-vessels-using-GES-net-)

and:

(Mask R-CNN)[]

# Vector LabPics dataset
The Vector LabPics dataset contains images of materials and vessels in the chemistry lab and other settings. The images are annotated for both the materials and vessels for both semantic and instance segmentation. Examples can be seen in Figures 1 and 2. The full dataset can be download from   [Here](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)
