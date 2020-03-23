import os
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms


class MedDataset(datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, transforms=None):
        super(MedDataset, self).__init__(root, transforms, transform, target_transform)
        self.examples = []
        self.root = root
        for sub_folder in os.listdir(self.root):
            for img in os.listdir(os.path.join(self.root, sub_folder)):
                self.examples.append(os.path.join(self.root, sub_folder, img))

    def __getitem__(self, idx):
        data_path = self.examples[idx]
        img = Image.open(data_path).convert("RGB")
        target = {"fname": data_path.split('\\')[-1]}
        if self.transforms is not None:
            img, target= self.transforms(img,target)
        return img, target

    def __len__(self):
        return len(self.examples)


class ChemScapeDataset(datasets.VisionDataset):
    def __init__(self, root, source, readEmpty=False, transform=None, target_transform=None, transforms=None, classes=None, coco=False):
        super(ChemScapeDataset, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        self.transforms = transforms
        self.source = source
        self.annotations = []
        self.readEmpty = readEmpty
        self.classes = set()
        print("Creating annotation list for reader this might take a while")
        avg_ids = 0
        for AnnDir in os.listdir(self.root):
            for SubDir in self.source:
                path = os.path.join(self.root, AnnDir, SubDir)
                if not os.path.isdir(path):
                    print("No folder:" + path)
                    continue
            if self.readEmpty:
                SubDirs = self.source+ ["EmptyRegions"]
            else:
                SubDirs = self.source
            CatDic = {}
            if coco:
                CatDic["Image"] = os.path.join(self.root, AnnDir, "Image.jpg")
            else:
                CatDic["Image"] = os.path.join(self.root, AnnDir, "Image.png")
            CatDic["folder"] = AnnDir
            CatDic["instances"] = []
            for sdir in SubDirs:
                obj = {}
                InstDir = os.path.join(self.root, AnnDir, sdir)
                if not os.path.isdir(InstDir): continue
                num_instances = 0
                # ------------------------------------------------------------------------------------------------
                for Name in os.listdir(InstDir):
                    num_instances += 1
                    CatString = ""
                    if "CatID_" in Name:
                        CatString = Name[Name.find("CatID_") + 6:Name.find(".png")]
                    ListCat = []
                    if sdir == "EmptyRegions": ListCat = [0]
                    while (len(CatString) > 0):
                        if "_" in CatString:
                            ID = int(CatString[:CatString.find("_")])
                        else:
                            ID = int(CatString)
                            CatString = ""
                        if not ID in ListCat: ListCat.append(ID)
                        CatString = CatString[CatString.find("_") + 1:]
                    obj["Cats"] = ListCat
                    obj["Ann"] = os.path.join(InstDir, Name)
                if num_instances == 0: continue
                CatDic["instances"].append(obj)
                avg_ids += len(obj["Cats"])
                self.classes.update(obj["Cats"])
            if len(CatDic["instances"]) == 0:
                print("No instance" + CatDic["Image"])
                continue
            self.annotations.append(CatDic)
        self.classes = sorted(self.classes)
        self.class_count = 16
        if classes is None:
            self.classes = {self.classes[i]:i+1 for i in range(len(self.classes))}
        else:
            self.classes = classes
        print(self.classes)
        print("Total=" + str(len(self.annotations)))
        print("avg classes per instance=" + str(avg_ids/len(self.annotations)))
        print("done making file list")

    def __getitem__(self, idx):
        data_path = self.annotations[idx]
        img = Image.open(data_path["Image"]).convert("RGB")

        num_objs = len(data_path["instances"])
        labels = []
        sub_class = []
        masks = []
        boxes = []
        for instance in data_path["instances"]:
            if "Cats" not in instance or len(instance["Cats"]) == 0:
                print(data_path["Image"])
                print(instance["Ann"])
            #get the superclass, convert to class index
            labels.append(self.classes[instance["Cats"][0]])
            sub_label = np.zeros(self.class_count+1)
            sub_label[instance["Cats"]] = 1
            #sub_label[[1,2,3,4,5]] = 0
            sub_class.append(sub_label)
            mask = Image.open(instance["Ann"])
            mask = np.array(mask)
            if len(mask.shape) == 3:
                mask = mask[:, :, -1]
            foreGround = (mask > 0) * (mask < 3)
            masks.append(foreGround)

            pos = np.where(foreGround)
            try:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            except:
                print(pos)
                print(data_path["Image"])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        sub_class = torch.as_tensor(sub_class, dtype=torch.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(list(boxes.size())) < 2:
            print(data_path["Image"])
            return img, {}
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if len(boxes[keep].tolist()) == 0 :
            print(boxes)
            print(data_path["Image"])
        boxes = boxes[keep]
        labels = labels[keep]
        sub_class = sub_class[keep]
        masks = masks[keep]
        if num_objs == 0:
            area = torch.zeros((num_objs,), dtype=torch.int64)
            print(data_path["Image"])
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "sub_cls":sub_class,
            #"fname": data_path["folder"]
        }
        if self.transforms is not None:
            img,target = self.transforms(img,target)
        return img, target

    def __len__(self):
        return len(self.annotations)


if __name__=="__main__":
    dataset = ChemScapeDataset(os.path.join("../../../ChemLabScapeDataset/Complex", "Train"), ["Vessel","Material"], transforms=None)
    for i in range(10, 1000, 100):
        img,target = dataset[i]
        print(target["sub_cls"])
        print(target["labels"])
