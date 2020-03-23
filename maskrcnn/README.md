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

`https://drive.google.com/file/d/1JBlyOiDK9NVhwra8Zs1WywJM3FvE6-1k/view?usp=sharing`


subclass with coco related dataset

`https://drive.google.com/file/d/1bbh4YDJuWU9bUe5PQm0KHVw-qGtxvKVn/view?usp=sharing`
