import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging

import open_clip
from dataset import VisaDataset, MVTecDataset
from cyhModule.model import LinearLayer, LinearLayerTuning

import cv2

'''''
prompt learning + tuning + loss
'''




def train(args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)



    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model = model.train()
    model.to(device)


    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets
    if args.dataset == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform,
                                  aug_rate=args.aug_rate)
    else:
        train_data = VisaDataset(root=args.train_data_path, transform=preprocess, target_transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)





    for items in train_dataloader:

        anomaly = items["anomaly"].item()
        if not anomaly:
            continue
        image_path = items['img_path'][0]
        mask_path = items["mask_path"][0]

        image = cv2.imread(image_path)
        image = cv2.resize(image, (240, 240))

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (240, 240))

        if args.dataset == 'visa':
            mask = mask * 255
        mask[:, :, :2] *= 0

        mask = mask.astype(image.dtype)

        rst = cv2.addWeighted(mask, 0.5, image, 0.5, 0)


        directory_name = os.path.dirname(image_path)
        save_directory_path = directory_name.replace('data', "ground_truth")
        if not os.path.exists(save_directory_path):
            os.makedirs(save_directory_path)

        save_path = image_path.replace('data', 'ground_truth')
        cv2.imwrite(save_path, rst)










if __name__ == '__main__':
    # parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # # path
    # parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    # parser.add_argument("--save_path", type=str, default='./exps/vit_b_16+', help='path to save results')
    # parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16-plus-240.json', help="model configs")
    # # model
    # parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    # parser.add_argument("--model", type=str, default="ViT-B-16-plus-240", help="model used")
    # # parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    #
    # parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    # parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6], help="features used")
    # # hyper-parameter
    # parser.add_argument("--epoch", type=int, default=50, help="epochs")
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    # # parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # parser.add_argument("--batch_size", type=int, default= 1, help="batch size")
    # parser.add_argument("--image_size", type=int, default=240, help="image size")
    # parser.add_argument("--aug_rate", type=float, default=-1, help="image size")
    # parser.add_argument("--print_freq", type=int, default=30, help="print frequency")
    # parser.add_argument("--save_freq", type=int, default=3, help="save frequency")
    # args = parser.parse_args()

    # vb16
    # parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # # path
    # # parser.add_argument("--train_data_path", type=str, default="../data/mvtec", help="train dataset path")
    # parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    #
    # parser.add_argument("--save_path", type=str, default='./exps/vit_large_14_518', help='path to save results')
    # parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json',
    #                     help="model configs")
    # # model
    # # parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    # parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    # parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    # # parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    # parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    # # hyper-parameter
    # parser.add_argument("--epoch", type=int, default=15, help="epochs")
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    # parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    # parser.add_argument("--image_size", type=int, default=224, help="image size")
    # parser.add_argument("--aug_rate", type=float, default=-1, help="image size")
    # parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    # parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    # args = parser.parse_args()
    #
    # setup_seed(111)
    # train(args)

    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/vit_b_16+', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16-plus-240.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-B-16-plus-240", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    # parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--batch_size", type=int, default= 1, help="batch size")
    parser.add_argument("--image_size", type=int, default=240, help="image size")
    parser.add_argument("--aug_rate", type=float, default=-1, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--print_eval", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    args = parser.parse_args()


    train(args)

    # vb14
    # parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # # path
    # parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    # parser.add_argument("--save_path", type=str, default='./exps/vit_b_16+', help='path to save results')
    # # parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16-plus-240.json', help="model configs")
    # parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    # # model
    # parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    # # parser.add_argument("--model", type=str, default="ViT-B-16-plus-240", help="model used")
    # parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    # parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    # parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    # # hyper-parameter
    # parser.add_argument("--epoch", type=int, default=15, help="epochs")
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    # # parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # parser.add_argument("--batch_size", type=int, default= 2, help="batch size")
    # # parser.add_argument("--image_size", type=int, default=240, help="image size")
    # parser.add_argument("--image_size", type=int, default=336, help="image size")
    # parser.add_argument("--aug_rate", type=float, default=-1, help="image size")
    # parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    # parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    # args = parser.parse_args()
    #
    # setup_seed(111)
    # train(args)

