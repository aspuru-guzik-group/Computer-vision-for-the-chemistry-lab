import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from Utils.coco_utils import get_coco, get_coco_kp

from Utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from Utils.engine import train_one_epoch, evaluate
from Reader.InstanceReader.InstanceReaderCoCoStyle import ChemScapeDataset
from Utils.coco_utils import get_coco_api_from_dataset
from Utils.coco_eval import CocoEvaluator
from Utils.engine import _get_iou_types
from Utils.utils import get_transform
import numpy as np
import cv2 as cv
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import tkinter
import matplotlib
matplotlib.use('TkAgg')

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    if cv.__version__.startswith('4'):
        contours, hierarchy = cv.findContours(*args, **kwargs)
    elif cv.__version__.startswith('3'):
        _, contours, hierarchy = cv.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


class ChemDemo(object):
    def __init__(self, model, data_loader, confidence_threshold=0.7, device=torch.device("cpu")):
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.confidence = confidence_threshold
        self.mask_threshold = 0.5
        self.iou_types = _get_iou_types(model)
        self.palette = [[0,0,255], [0,255,0], [255,0,0], [255,255,0], [0,255,255], [255,0,255], [255,255,255]]

    def compute_prediction(self, image):
        cpu_device = torch.device("cpu")
        image = list(img.to(self.device) for img in image)
        outputs = self.model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        return outputs[0]

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions["scores"]
        keep = torch.nonzero(scores > self.confidence).squeeze(1)
        predictions = {key: predictions[key][keep] for key in predictions.keys()}
        return predictions

    def run_on_image(self, image, target, outDir):
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        results = image[0].numpy().copy()
        results = np.dstack((results[0], results[1], results[2]))
        results = self.overlay_boxes(results, top_predictions)
        results = self.overlay_mask(results, top_predictions)
        plt.imshow(results)
        #plt.show()
        plt.savefig(outDir + "/" + target[0]["fname"] + '.png')
        plt.clf()
        return results

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions["labels"]
        boxes = predictions["boxes"]
        scores = predictions["scores"]
        template = "{}: {:.2f}"
        for label, box, score in zip(labels, boxes, scores):
            color = self.compute_colors_for_labels(label)
            box = box.to(torch.int64)
            x, y = box[2:]
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 1)
            s = template.format(label, score)
            image = cv.putText(
                image, s, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1
            )
        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions["masks"].numpy()
        labels = predictions["labels"]

        for mask, label in zip(masks, labels):
            color = self.compute_colors_for_labels(label)
            contours, hierarchy = findContours(((mask[0]>0.50)*255).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            image = cv.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def compute_colors_for_labels(self, label):
        """
        Simple function that adds fixed colors depending on the class
        """
        color = self.palette[label]
        return color

    def compute_panoptic(self, image, label, folder, json, show=False):
        sub_classes = ["",
                       "V",
                       "V Label",
                       "V Cork",
                       "V Parts GENERAL",
                       "Ignore",
                       "Liquid GENERAL",
                       "Liquid Suspension",
                       "Foam",
                       "Gel",
                       "Solid GENERAL",
                       "Granular",
                       "Powder",
                       "Solid Bulk",
                       "Vapor",
                       "Other Material",
                       "Filled vessel"]
        json[label[0]["fname"]] = {
            "PartCats": {},
            "MaterialCats": {},
            "MultiPhaseMaterial": [],
            "MultiPhaseVessels": []
        }
        material_cat = {}
        predictions = self.compute_prediction(image)
        top_pred = self.select_top_predictions(predictions)
        material = np.zeros((image[0].size(1), image[0].size(2)))
        vessel = material.copy()
        parts = material.copy()
        m_id = 1
        v_id = 1
        for i in range(top_pred["scores"].size(0)):
            if top_pred["labels"][i] not in [0,2]:
                sub_cls = torch.nonzero(top_pred['sub_cls'][i]).squeeze(1).tolist()
                sub_labels = [sub_classes[idx] for idx in sub_cls]
                material_cat[str(m_id)]= sub_labels
                material += (top_pred['masks'][i][0].numpy() > 0.5) * (material == 0) * m_id
                m_id += 1
            if top_pred["labels"][i] == 2:
                sub_cls = torch.nonzero(top_pred['sub_cls'][i]).squeeze(1).tolist()
                sub_labels = [sub_classes[idx] for idx in sub_cls]
                material_cat[str(m_id)] = sub_labels
                vessel += (top_pred['masks'][i][0].numpy() > 0.5) * (vessel == 0) * v_id
                v_id += 1

        if show:
            image = image[0].numpy().copy()
            image = np.dstack((image[0], image[1], image[2]))
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            ax[1].matshow(material)
            ax[2].matshow(vessel)
            plt.show()

        anno_file = os.path.join(folder, label[0]["fname"])
        results = np.dstack(( vessel,parts, material)).astype(np.uint8)
        plt.imsave(anno_file+".png", results)

        json[label[0]["fname"]]["MaterialCats"] = material_cat
        return json
