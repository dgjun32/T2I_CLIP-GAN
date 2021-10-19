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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import transformers

from masks import mask_correlated_samples_2
from nt_xent import NT_Xent

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
    parser.add_argument('--clip_cfg', dest = 'clip_cfg', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(dataloader, clip, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir, criterion):
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
        sort_ind, sort_ind_2 = prepare_data(data) # data.cuda()

        
        # extract image and text features
        sent_code, subr_feature, sent_emb, words_emb = clip(imgs[0], captions)
        sent_code_2, subr_feature_2, sent_emb_2, words_emb_2 = clip(imgs_2[0], captions_2)
        
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
        torch.nn.utils.clip_grad_norm(clip.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)

        optimizer.step()

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
    return count


def evaluate(dataloader, clip, batch_size, criterion):
    clip.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        # real_imgs, captions, cap_lens, \
        #         class_ids, keys = prepare_data(data)

        imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
        sort_ind, sort_ind_2 = prepare_data(data)

        with torch.no_grad():
            # extract image and text features
            sent_code, subr_feature, sent_emb, words_emb = clip(imgs[0], captions)
            
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


def build_models(state_dict):
    # build model ############################################################
    
    clip = build_clip(state_dict)

    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    print('start_epoch', start_epoch)
    if cfg.CUDA:
        clip = clip.cuda()
        labels = labels.cuda()

    return clip, labels, start_epoch


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
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '/home/coder/dongjun/CLIP+GAN/AttnGAN+CLIP/output/%s_%s/' % \
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
    dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Build model ##############################################################
    cfg.PRETRAIN_DIR = '../pretrained_clip/{}'.format(args.clip_cfg)
    state_dict = torch.load(cfg.PRETRAIN_DIR)
    clip, labels, start_epoch = build_models(state_dict)
    para = list(clip.parameters())

    # Train ##############################################################
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    mask = mask_correlated_samples_2(batch_size)

    temperature = 0.5
    device = labels.get_device()
    criterion = NT_Xent(batch_size, temperature, mask, device)


    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, clip,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir, criterion)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, clip, batch_size, criterion)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(clip.state_dict(),
                           '%s/clip%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
