from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

import os
import sys
import time
import random
import math
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image
import tensorboardX

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import transformers

from masks import mask_correlated_samples_2
from nt_xent import NT_Xent

# pretrained CLIP
from clip.model import CLIP
from clip.model import build_clip
from clip.clip_api import tokenize

# Logger
from tensorboardX import SummaryWriter

summary = SummaryWriter()
UPDATE_INTERVAL = 50

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(dataloader, clip, batch_size,
          labels, backbone_optimizer, linear_optimizer, epoch, ixtoword, image_dir, criterion, tokenizer):
    clip.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    

    for step, data in enumerate(dataloader, 0):
        # data : imgs = [image tensor], caps, caps_len, cls_id, key, caps_2, caps_len_2

        # print('step', step)
        clip.zero_grad()

        # imgs, captions, cap_lens, \
        #     class_ids, keys = prepare_data(data)

        imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
        sort_ind, sort_ind_2 = prepare_data(data, tokenizer) # data.cuda()
        '''
        imgs : list containing image tensor
        captions : dict containig input_ids and attention_mask
        '''
        # extract image and text features
        sent_code, subr_feature, sent_emb, words_emb = clip(**captions, pixel_values = imgs[0])
        sent_code_2, subr_feature_2, sent_emb_2, words_emb_2 = clip(**captions_2, pixel_values = imgs_2[0])
        
        # tensor size
        nef = subr_feature.shape[2]
        att_sze = int(math.sqrt(subr_feature.shape[1] - 1))
        seq_len = words_emb.shape[1]
        batch_size = words_emb.shape[0]

        # transform tensors
        words_features = subr_feature[:,1:,:].permute(0,2,1).reshape(batch_size, nef, att_sze, att_sze)
        words_features_2 = subr_feature[:,1:,:].permute(0,2,1).reshape(batch_size, nef, att_sze, att_sze)
        words_emb = words_emb.permute(0,2,1)
        words_emb_2 = words_emb_2.permute(0,2,1)
        
        # compute loss
        ## word - subregion level attention loss
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        w2_loss0, w2_loss1, attn_maps_2 = words_loss(words_features_2, words_emb_2, labels,
                                                 cap_lens_2, class_ids_2, batch_size)
        w_total_loss0 += w2_loss0.data
        w_total_loss1 += w2_loss1.data
        loss += w2_loss0 + w2_loss1

        ## sentence - image level attention loss
        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        s2_loss0, s2_loss1 = \
            sent_loss(sent_code_2, sent_emb_2, labels, class_ids_2, batch_size)
        loss += s2_loss0 + s2_loss1
        s_total_loss0 += s2_loss0.data
        s_total_loss1 += s2_loss1.data

        _, ori_indices = torch.sort(sort_ind, 0)
        _, ori_indices_2 = torch.sort(sort_ind_2, 0)

        sent_emb = sent_emb[ori_indices]
        sent_emb_2 = sent_emb_2[ori_indices_2]

        # sent_emb = l2norm(sent_emb, dim=1)

        sent_emb = l2norm(sent_emb, dim=1)
        sent_emb_2 = l2norm(sent_emb, dim=1)

        contrative_loss = criterion(sent_emb, sent_emb_2)
        loss += contrative_loss



        #

        # mse_loss = nn.MSELoss(reduction='sum')
        # q_out = netD.Q_NET(fake_features)
        # l2_loss = mse_loss(sent_code, sent_emb)
        # batch_size = region_features.size(0)
        # l2_loss = l2_loss / batch_size
        # l2_loss = l2_loss * 0.1
        # print(l2_loss)

        # loss += l2_loss

        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(clip.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)

        backbone_optimizer.step()
        linear_optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            # img_set, _ = \
            #     build_super_images(imgs[-1].cpu(), captions,
            #                        ixtoword, attn_maps, att_sze)
            # if img_set is not None:
            #     im = Image.fromarray(img_set)
            #     fullpath = '%s/attention_maps%d.png' % (image_dir, step)
            #     im.save(fullpath)
    return count, loss


def evaluate(dataloader, clip, batch_size, criterion, tokenizer):
    clip.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        # real_imgs, captions, cap_lens, \
        #         class_ids, keys = prepare_data(data)

        imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
        sort_ind, sort_ind_2 = prepare_data(data, tokenizer)

        with torch.no_grad():
            # extract image and text features
            sent_code, subr_feature, sent_emb, words_emb = clip(**captions, pixel_values = imgs[0])
            
            # tensor size
            nef = subr_feature.shape[2]
            att_sze = int(math.sqrt(subr_feature.shape[1] - 1))
            seq_len = words_emb.shape[1]
            batch_size = words_emb.shape[0]

            # transform tensors
            words_features = subr_feature[:,1:,:].permute(0,2,1).reshape(batch_size, nef, att_sze, att_sze)
            words_emb = words_emb[:,1:,:].permute(0,2,1)

            # calculate loss
            w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                                cap_lens, class_ids, batch_size)
            w_total_loss += (w_loss0 + w_loss1).data

            s_loss0, s_loss1 = \
                sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss

class AddLinearOnCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.linear_subr = nn.Linear(768, 512)
    def forward(self, pixel_values, input_ids, attention_mask):
        batch_size = pixel_values.shape[0]
        outputs = self.backbone(pixel_values = pixel_values, input_ids = input_ids, attention_mask = attention_mask)
        img, subr = outputs['image_embeds'], outputs['vision_model_output']['last_hidden_state']
        sent, words = outputs['text_embeds'], outputs['text_model_output']['last_hidden_state']
        # linear transformation for same embedding dimension -> compute word loss
        subr = self.linear_subr(subr.view(-1,768)).view(batch_size,-1,512)
        return img, subr, sent, words

def build_models():
    # build model ############################################################
    clip = AddLinearOnCLIP()
    tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    print('start_epoch', start_epoch)
    if cfg.CUDA:
        clip = clip.cuda()
        labels = labels.cuda()

    return clip, labels, start_epoch, tokenizer

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################

    output_dir = '/home/coder/dongjun/CLIP+GAN/DMGAN+CLIP/output/%s_%s/' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'val',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Build model ##############################################################
    clip, labels, start_epoch, tokenizer = build_models()
    backbone_para = list(clip.backbone.parameters())
    linear_subr_para = list(clip.linear_subr.parameters())
    # Train ##############################################################
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    mask = mask_correlated_samples_2(batch_size)

    temperature = 0.5
    device = labels.get_device()
    criterion = NT_Xent(batch_size, temperature, mask, device)


    try:
        backbone_lr = cfg.TRAIN.BACKBONE_LR
        linear_lr = cfg.TRAIN.LINEAR_LR
        # optimizer
        backbone_optimizer = optim.AdamW(backbone_para, lr = 1e-7, betas = (0.5, 0.999))
        linear_optimizer = optim.AdamW(linear_subr_para, lr = 1e-7, betas = (0.5, 0.999))
        # lr schedule
        backbone_sched = CosineAnnealingWarmUpRestarts(backbone_optimizer, T_0=3, T_mult=1,
                                                     eta_max=backbone_lr, T_up=1, gamma=0.5)
        linear_sched = CosineAnnealingWarmUpRestarts(linear_optimizer, T_0 =3, T_mult=1,
                                                     eta_max=linear_lr, T_up=1, gamma=0.5)
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            epoch_start_time = time.time()
            count, wsc_loss = train(dataloader, clip,
                          batch_size, labels, backbone_optimizer, linear_optimizer, epoch,
                          dataset.ixtoword, image_dir, criterion, tokenizer)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, clip, batch_size, criterion, tokenizer)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | backbone_lr {:.5f}| linear_lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, backbone_lr, linear_lr))
            print('-' * 89)
            # val_loss logging
            summary.add_scalar('train_loss/wsc_loss', wsc_loss.item(), epoch)
            summary.add_scalar('val_loss/w_loss', w_loss, epoch)
            summary.add_scalar('val_loss/s_loss', s_loss, epoch)
            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(clip.state_dict(),
                           '%s/clip%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
            backbone_sched.step()
            linear_sched.step()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')