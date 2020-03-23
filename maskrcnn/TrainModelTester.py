from Reader.InstanceReader.InstanceReaderCoCoStyle import ChemScapeDataset
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
from torchvision import datasets, transforms

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)
mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=14,sampling_ratio=2)
model = MaskRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

dataDir= "../ChemLabScapeDataset/TrainAnnotations"
#dataset = ChemScapeDataset(dataDir, None, "Vessel", False)
d = datasets.CocoDetection(root="../coco/train2014", annFile="../coco/annotations/instances_train2014.json", transform=transforms.ToTensor())
dataLoader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True, num_workers=0)

for batch_idx, (data, target) in enumerate(dataLoader):
    print(data.size())
    print(target)
    model(data, target)
    break

