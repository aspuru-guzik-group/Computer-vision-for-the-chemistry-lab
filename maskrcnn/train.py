r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
"""
import datetime
import os
import time

import torch
from torch import nn
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads

from Utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from Utils.engine import train_one_epoch, evaluate

from Utils import utils
from Reader.InstanceReader.InstanceReaderCoCoStyle import ChemScapeDataset, MedDataset
import numpy as np
from Utils.Visual import ChemDemo
import detection
import json

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    for dataset in args.dataset:
        if dataset not in ['Vessel', 'Material']:
            print(dataset + " not in listed dataset")
            exit(1)
    #dataset, num_classes = utils.get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    #dataset_test, _ = utils.get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
    # classes = {1:2, 2:2, 3:2, 4:2, 5:0, 6:1, 7:1, 8:1, 9:1, 10:3, 11:3, 12:3, 13:3, 14:4, 15:3, 16:3}
    # dataset = ChemScapeDataset(os.path.join(args.data_path, "Train"), args.dataset,
    #                            transforms=utils.get_transform(train=not args.test_only), classes=classes)
    # dataset_test = ChemScapeDataset(os.path.join(args.data_path, "Test"), args.dataset,
    #                                 transforms=utils.get_transform(train=False), classes=dataset.classes)
    # coco_dataset = ChemScapeDataset("../ChemLabScapeDataset/COCO_2017_related_complex", ['Vessel'],
    #                                 transforms=utils.get_transform(train=not args.test_only), classes=classes, coco=True)
    dataset = MedDataset(args.data_path, transforms=utils.get_transform(train=False))
    dataset_test = dataset
    coco_dataset = dataset
    num_classes = 5
    # print("Dataset {}, num_class {}, class_list {}".format(args.dataset, num_classes, dataset.classes))
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        coco_sampler = torch.utils.data.distributed.DistributedSampler(coco_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        coco_sampler = torch.utils.data.RandomSampler(coco_dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        coco_batch_sampler = GroupedBatchSampler(coco_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)
        coco_batch_sampler = torch.utils.data.BatchSampler(
            coco_sampler, args.batch_size, drop_last=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    coco_data_loader = torch.utils.data.DataLoader(
        coco_dataset, batch_sampler=coco_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    print("Creating model")
    if args.subclass:
        print("predicting subclasses")
        model = detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained, num_sub_cls=17)
    else:
        model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        print("loading trained model")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        train_eval_sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader_train = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            sampler=train_eval_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)
        with torch.no_grad():
            for t in np.arange(0.55, 0.75, 0.05):
                demo = ChemDemo(model, data_loader_test, device=device, confidence_threshold=t)
                if not os.path.exists(args.output_dir + "/testAnno{}".format(t)):
                    os.makedirs(args.output_dir +"/testAnno{}".format(t))
                if not os.path.exists(args.output_dir + "/trainAnno{}".format(t)):
                    os.makedirs(args.output_dir + "/trainAnno{}".format(t))
                train_json = {}
                test_json = {}
                for batch_idx, (image, target) in enumerate(data_loader_test):
                    demo.run_on_image(image, target, args.output_dir + "/testAnno{}".format(t))
                    #test_json = demo.compute_panoptic(image, target, args.output_dir +"/testAnno{}".format(t), test_json)

                # with open(args.output_dir + "/testAnno{}/".format(t) + 'test.json', 'w') as f:
                #     json.dump(test_json, f)

                # for batch_idx, (image, target) in enumerate(data_loader_train):
                #     demo.run_on_image(image, target,args.output_dir + "/trainAnno{}".format(t))
                #     train_json = demo.compute_panoptic(image, target, args.output_dir +"/trainAnno{}".format(t), train_json)

                # with open(args.output_dir + "/trainAnno{}/".format(t) + 'train.json', 'w') as f:
                #     json.dump(train_json, f)
            #evaluate(model,data_loader_train, device=device)
            #evaluate(model, data_loader_test, device=device)
            return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch % 2 == 1:
            train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        else:
            print(f"{len(coco_dataset)} coco dataset")
            train_one_epoch(model, optimizer, coco_data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir and epoch % 10 == 0:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='../ChemLabScapeDataset/MedicalImages', help='dataset')
    parser.add_argument('--dataset',nargs='*', default=['Vessel'], help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02/8, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=10, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=-1, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--subclass",
        help="predict subclasses",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--distributed",
        help="Use distributed gpu to train models",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
