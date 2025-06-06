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
from loss import FocalLoss, BinaryDiceLoss, Cross_Entropy_Loss, Discriminability_prompt_loss, Variance_Loss

from cyhModule import coop, linear

'''''
prompt learning + loss
'''

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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, model_configs, device):
        super().__init__()
        self.prompt_learner = coop.PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.model = clip_model
        self.features_list = list(range(1,10))
        # self.features_list = [3, 6, 9]

        self.text_prompt_layer = linear.Text_prompt_layer().to(device)

        self.location_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(self.features_list), args.model).to(device)

        self.tuning_layer = LinearLayerTuning().to(device)
        self.tokenizer = open_clip.get_tokenizer(args.model)
        self.device = device
        self.logit_scale = clip_model.logit_scale



    def forward(self, image, auxiliary_token_embedding):
        image_features, patch_tokens = self.model.encode_image(image, self.features_list)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        image_prompt = self.text_prompt_layer(patch_tokens)
        image_prompt = torch.unsqueeze(image_prompt, dim=-1)
        prompts = self.prompt_learner()
        auxiliary_token_embedding = self.tuning_layer(auxiliary_token_embedding)
        prompts = prompts + image_prompt + auxiliary_token_embedding
        tokenized_prompts = self.tokenized_prompts
        text_features = self.model.encode_text(tokenized_prompts, prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logits = (100 * image_features @ text_features.t()).softmax(dim=-1)

        # pixel level
        patch_tokens = self.location_layer(patch_tokens)
        anomaly_maps = []
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)  # 归一化
            anomaly_map = (100.0 * patch_tokens[layer] @ text_features.t())
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=args.image_size, mode='bilinear', align_corners=True)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_maps.append(anomaly_map)

        return logits, prompts, anomaly_maps


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

    loss_of_train = Loss_figure()


    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model = model.train()
    model.to(device)


    cudnn.benchmark = True


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


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_cross_entropy = Cross_Entropy_Loss()
    loss_discriminability_prompt = Discriminability_prompt_loss()
    loss_var_px = Variance_Loss()

    tokenizer = open_clip.get_tokenizer(args.model)

    def get_tuning_prompt():
        auxiliary_token_embedding = ['small']

        auxiliary_token_embedding = tokenizer(auxiliary_token_embedding).to(device)
        auxiliary_token_embedding = model.encoder_tuning(auxiliary_token_embedding)


        return auxiliary_token_embedding.detach()

    auxiliary_token_embedding = get_tuning_prompt()

    # #1
    class_name = ["normal object", "abnormal object"]
    clip_model = CustomCLIP(class_name, model, model_configs, device)


    for name, params in clip_model.named_parameters():
        if 'ctx' not in name and 'text_prompt_layer' not in name and 'location_layer' not in name and 'tuning_layer' not in name:
            params.requires_grad = False


    optimizer = torch.optim.Adam(clip_model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)


    for epoch in range(epochs):
        loss_list = []
        idx = 0
        for items in train_dataloader:
            idx += 1
            image = items['img'].to(device)
            cls_name = items['cls_name']
            ground_truth = items['anomaly']

            text_probs, prompt, anomaly_maps = clip_model(image, auxiliary_token_embedding)

            # loss
            loss_DP = loss_discriminability_prompt(prompt)
            loss_CE = loss_cross_entropy(text_probs, ground_truth)
            loss =  loss_CE + 0.05 * loss_DP
            # loss =  loss_CE


            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0  # 映射到0或1的二值图像

            for num in range(len(anomaly_maps)):
                loss += 0.005 * loss_focal(anomaly_maps[num], gt)  # 单独每一层训练，这样让为0的通道都为0，为1的通道都为1，这样效果应该更好一些（猜测，作者没说，应该实验之后，发现这样性能更好）
                loss += 0.005 * loss_dice(anomaly_maps[num][:, 1, :, :], gt)  # 把表示正常的层提取出来或异常的，到时候看一下dice loss
                loss += 0.005 * loss_var_px(anomaly_maps[num][:, 1, :, :], gt) / 9

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
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': clip_model.state_dict()}, ckp_path)

        # # eval
        # if epoch == args.print_evel:
        #     auc = auc_eval(args)

    loss_of_train.plot_figure()



if __name__ == '__main__':
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

    setup_seed(111)
    train(args)


