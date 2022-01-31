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
import cv2
import tensorboardX
import matplotlib.pyplot as plt

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
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def rm_special_token(mask, words_emb):
    '''
    words_emb : torch.FloatTensor shape of (batch_size, 30, 512)
    mask : torch.BoolTensor shape of (batch_size, 30)
    '''
    batch_size, n_words, emb = words_emb.size()
    words_emb_new, mask_new = [], []
    for i in range(batch_size):
        if torch.sum(mask[i]) == n_words:
            emb = words_emb[i,1:-1,:]
            m = mask[i,1:-1]
        else:
            eos_idx = torch.where(mask[i]==0)[0].min()
            emb = words_emb[i]
            emb = torch.cat([emb[1:(eos_idx-1),:], emb[eos_idx:]], dim=0)
            m = mask[i]
            m = torch.cat([m[1:(eos_idx-1)], m[eos_idx:]], dim=0)
        words_emb_new.append(emb)
        mask_new.append(m)
    words_emb_new = torch.stack(words_emb_new, dim=0)
    mask_new = torch.stack(mask_new, dim=0)
    return words_emb_new, mask_new    


def train(dataloader, clip, batch_size, labels,
        backbone_optimizer, linear_optimizer,
        epoch, ixtoword, image_dir, criterion, tokenizer):
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
        sort_ind, sort_ind_2 = prepare_data(data, tokenizer, words_num=30) # data.cuda()
        '''
        imgs : list containing image tensor
        captions : dict containig input_ids and attention_mask
        '''

        # words_mask
        words_mask = captions['attention_mask']
        words_mask_2 = captions_2['attention_mask']

        # extract image and text features
        sent_code, subr_feature, sent_emb, words_emb = clip(**captions, pixel_values = imgs[0])
        sent_code_2, subr_feature_2, sent_emb_2, words_emb_2 = clip(**captions_2, pixel_values = imgs_2[0])
        
        # tensor size
        nef = subr_feature.shape[2]
        att_sze = int(math.sqrt(subr_feature.shape[1] - 1))
        seq_len = words_emb.shape[1]
        batch_size = words_emb.shape[0]

        # transform tensors
        ## remove <sos> image token
        words_features = subr_feature[:,1:,:].permute(0,2,1) # shape of (batch_size, emb, n_patches)
        words_features_2 = subr_feature[:,1:,:].permute(0,2,1)
        ## remove <sos>,<eos> image token from words_emb, words_mask
        words_emb, words_mask = rm_special_token(words_mask, words_emb)
        words_emb_2, words_mask_2 = rm_special_token(words_mask_2, words_emb_2)
        words_emb = words_emb.permute(0,2,1) # shape of (batch_size, emb, n_words)
        words_emb_2 = words_emb_2.permute(0,2,1)
        
        # compute loss
        ## word - subregion level attention loss
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                cap_lens, class_ids, batch_size,
                                                words_mask, cfg.TRAIN.SMOOTH.GAMMA1, cfg.TRAIN.SMOOTH.GAMMA2, cfg.TRAIN.SMOOTH.GAMMA3)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        w2_loss0, w2_loss1, attn_maps_2 = words_loss(words_features_2, words_emb_2, labels,
                                                 cap_lens_2, class_ids_2, batch_size,
                                                 words_mask_2, cfg.TRAIN.SMOOTH.GAMMA1, cfg.TRAIN.SMOOTH.GAMMA2, cfg.TRAIN.SMOOTH.GAMMA3)
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

def build_super_images3(epoch, input_images, captions, attn_maps, tokenizer):
    '''
    input_images : normalized torch.FloatTensor shape of (batch_size, 3, 224, 224)
    captions : torch.LongTensor shape of (batch_size, 30)
    attn_maps : torch.FloatTensor shape of (batch_size, 49, 28)
    '''
    batch_size, n_patches, n_words = attn_maps.shape
    batch_size, _, img_size, img_size = input_images.shape
    n_grid = int(math.sqrt(n_patches))
    captions = captions[:,1:]
    inv_norm = transforms.Compose([
        transforms.Normalize(mean=[0.,0.,0.], std=[1/0.26862954, 1/0.26130258, 1/0.27577711]),
        transforms.Normalize(mean=[-0.48145466, -0.4578275, -0.40821073], std=[1.,1.,1.])
    ])

    fig = plt.figure(figsize = (n_words, batch_size*1.5))
    i = 1
    for b in range(batch_size):
        im = inv_norm(input_images[b]).permute(1,2,0).to('cpu').numpy()
        #im = np.uint8(255*im)
        for w in range(n_words):
            word = tokenizer.decode(captions[b,w]).encode('ascii','ignore').decode('ascii')
            if word == '<|endoftext|>':
                wd = '!'
            else:
                wd = word
            attn_map = attn_maps[b,:,w].reshape(n_grid, n_grid).unsqueeze(0).unsqueeze(0)
            attn_map = nn.Upsample(scale_factor=32)(attn_map)
            attn_map = attn_map[0].permute(1,2,0).to('cpu').numpy()
            # attn_map = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
            result = attn_map + im
            # result = (result-np.min(result)) / (np.max(result)-np.min(result))
            # drawing img+attnmap on figure
            plt.subplot(batch_size, n_words, i)
            plt.imshow(result)
            plt.title(wd)
            plt.xticks([])
            plt.yticks([])
            i+=1
    return fig


def evaluate(dataloader, clip, batch_size, criterion, tokenizer, epoch, cfg):
    clip.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        # real_imgs, captions, cap_lens, \
        #         class_ids, keys = prepare_data(data)

        imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
        sort_ind, sort_ind_2 = prepare_data(data, tokenizer, words_num=30)

        words_mask = captions['attention_mask'].cuda()

        with torch.no_grad():
            # extract image and text features
            sent_code, subr_feature, sent_emb, words_embs = clip(**captions, pixel_values = imgs[0])

            region_features = subr_feature[:,1:,].permute(0,2,1)
            region_features = subr_feature[:,1:,].permute(0,2,1)                
            words_embs, words_mask = rm_special_token(words_mask, words_embs) 
            words_embs = words_embs.permute(0,2,1) # shape of (batch_size, emb, n_words)

            ######################## compute attention map for visualization ############
            if step <= 0:
                mask = words_mask.unsqueeze(2)
                """
                words_emb(query): batch_size x emb_size x n_words 
                words_mask: batch_size x n_words x 1
                region_features(context): batch_size x emb_size x n_patches
                """
                batch_size, emb_size, n_words = words_embs.size()
                batch_size, emb_size, n_patches = region_features.size()
                assert mask.size() == (batch_size, n_words, 1) # batch_size, 28, 1
                # compute attnetion scores
                contextT = torch.transpose(region_features, 1, 2).contiguous()     
                queryT = torch.transpose(words_embs, 1, 2).contiguous()
                contextT = l2norm(contextT, dim=2)
                queryT = l2norm(queryT, dim=2)
                sim_scores = torch.bmm(queryT, torch.transpose(contextT, 1, 2))
                assert sim_scores.size() == (batch_size, n_words, n_patches)
                '''NOT FROM PAPER
                We should be masking out the similarity scores corresponding to padding tokens.
                
                    sim_scores.shape == [batch_size, n_words, n_patches]
                    words_mask.shape == [batch_size, n_words, 1]
                '''
                # masking attention values corresponding to padding token
                sim_scores = sim_scores.masked_fill_(mask == 0, -1)
                # applying softmax to sim_scores
                sm_sim_scores = torch.transpose(sim_scores, 1,2)
                ################################################################################
                fig = build_super_images3(epoch, imgs[0].cuda(), captions['input_ids'], sm_sim_scores, tokenizer)
                fig.savefig('/home/coder/dongjun/CLIP+GAN/DMGAN+CLIP/output/{}_DAMSM_img/attn_img_epoch{}.png'.format(cfg.DATASET_NAME, epoch+1))


            # calculate loss
            w_loss0, w_loss1, attn = words_loss(region_features, words_embs, labels,
                                                cap_lens, class_ids, batch_size,
                                                words_mask, cfg.TRAIN.SMOOTH.GAMMA1, cfg.TRAIN.SMOOTH.GAMMA2, cfg.TRAIN.SMOOTH.GAMMA3)
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
        outputs = self.backbone(pixel_values = pixel_values.cuda(),
                            input_ids = input_ids.cuda(),
                            attention_mask = attention_mask.cuda())
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

    output_dir = './output/%s_%s/' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # setting dataloader
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

    # At any point you can hit Ctrl + C to break out of training early.
    mask = mask_correlated_samples_2(batch_size)

    temperature = 0.5
    device = labels.get_device()
    criterion = NT_Xent(batch_size, temperature, mask, device)

    try:
        # opimizer hyperparams
        backbone_lr = cfg.TRAIN.BACKBONE_LR
        linear_lr = cfg.TRAIN.LINEAR_LR
        step_size_up = cfg.TRAIN.STEP_SIZE_UP
        gamma = cfg.TRAIN.GAMMA
        # optimizer
        backbone_optimizer = optim.Adam(backbone_para, lr = backbone_lr, betas = (0.9, 0.98))
        linear_optimizer = optim.Adam(linear_subr_para, lr = linear_lr, betas = (0.9, 0.98))
        # lr schedule
        backbone_sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=backbone_optimizer,
                                                    pct_start=0.02,
                                                    max_lr=backbone_lr,
                                                    anneal_strategy='cos',
                                                    cycle_momentum=False,
                                                    steps_per_epoch=len(dataloader),
                                                    epochs = cfg.TRAIN.MAX_EPOCH)
        linear_sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=linear_optimizer,
                                                    pct_start=0.1,
                                                    max_lr=linear_lr,
                                                    anneal_strategy='cos',
                                                    cycle_momentum=False,
                                                    steps_per_epoch=len(dataloader),
                                                    epochs = cfg.TRAIN.MAX_EPOCH,
                                                    div_factor=1e+3,
                                                    final_div_factor=1e+6)

        # start training
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            epoch_start_time = time.time()
            count, wsc_loss = train(dataloader, clip,
                            batch_size, labels, backbone_optimizer, linear_optimizer,
                            epoch, dataset.ixtoword, image_dir, criterion, tokenizer)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, clip, batch_size, criterion, tokenizer, epoch, cfg)
                print('| end epoch {:3d} | valid loss '
                        '{:5.2f} {:5.2f} | backbone_lr {:.8f}| linear_lr {:.5f}|'
                        .format(epoch, s_loss, w_loss, backbone_sched.get_last_lr()[0], linear_sched.get_last_lr()[0]))
            print('-' * 89)
            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(clip.state_dict(),
                            '%s/clip%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
            linear_sched.step()
            backbone_sched.step()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')