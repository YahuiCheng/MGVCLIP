import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging

import open_clip
from dataset import VisaDataset, MVTecDataset
from loss import Cross_Entropy_Loss

"""
The program for training the MSCA
"""

class Loss_figure():
    def __init__(self):
        super().__init__()

        self.train_loss = []
        self.save_path = 'loss_figure/loss.jpg'

    def plot_figure(self):
        plt.Figure()
        num_loss = len(self.train_loss)
        x_label = [i for i in range(num_loss)]
        plt.plot(x_label, self.train_loss)

        plt.savefig(self.save_path)
        # plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    loss_of_train = Loss_figure()

    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model = model.train()
    model.to(device)


    torch.backends.cudnn.benchmark = True

    tokenizer = open_clip.get_tokenizer(args.model)

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
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


    DATA_TOTAL_NUM = len(train_data)
    ITERATION_PER_EPOCH = DATA_TOTAL_NUM // batch_size


    for name, paras in model.named_parameters():
        if "multi_scale_adpter" not in name:
            paras.requires_grad = False


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    # losses
    # loss_focal = FocalLoss()
    # loss_dice = BinaryDiceLoss()
    loss_cross_entropy = Cross_Entropy_Loss()


    with torch.no_grad():
        text_prompt = ["normal object",
                       "abnormal object",]

        text_prompt = tokenizer(text_prompt).to(device)
        text_prompt = model.encode_text(text_prompt)
        text_prompt /= text_prompt.norm(dim=-1, keepdim=True) #[2, *]

    for epoch in range(epochs):
        loss_list = []
        idx = 0
        for items in train_dataloader:
            idx += 1
            image = items['img'].to(device)
            ground_truth = items['anomaly']

            # [N, *]
            image_features, patch_tokens = model.encode_image(image, features_list)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = (100 * image_features @ text_prompt.t()).softmax(dim=-1)  # [N, 1]

            loss = loss_cross_entropy(text_probs, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            loss_of_train.train_loss.append(np.mean(loss_list))

            logger.info('Epoch: {} | {}  Iteration: {} | {}  loss: {}'.format(epoch+1, epochs, idx, ITERATION_PER_EPOCH, np.mean(loss_list)))
        lr_scheduler.step()

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) +"_" + str(np.mean(loss_list)) + '.pth')
            torch.save({'trainable_linearlayer': model.state_dict()}, ckp_path)

    loss_of_train.plot_figure()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/vit_b_16+', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16-plus-240.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-B-16-plus-240", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=5, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    # parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--batch_size", type=int, default= 16, help="batch size")
    parser.add_argument("--image_size", type=int, default=240, help="image size")
    parser.add_argument("--aug_rate", type=float, default=-1, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    args = parser.parse_args()

    setup_seed(111)
    train(args)

