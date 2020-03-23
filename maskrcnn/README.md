#Mask RCNN with subclass head
environment requirement 
```
conda env create -f envt.yaml
```

training script 

``` 
python train.py --data-path [dataset path] --dataset Material Vessel --output-dir [training log path]  --subclass
```

Eval script
```
python train.py --data-path [dataset path] --dataset Material Vessel  --subclass --test-only --resume [model path] --output-dir [prediction result path]
```

##pretrained models
subclass without coco dataset

`maskrcnn/VML_Sub_COCO_R50_S20_G.1_A.0025_HVC_model_400.pth`


subclass with coco related dataset

`maskrcnn/VML_Sub_R50_S10_G.1_A.0025_HVC_model_190.pth`
