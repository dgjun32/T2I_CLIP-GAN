from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import math
import numpy as np
import sys

from masks import mask_correlated_samples
from nt_xent import NT_Xent


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, dataset, clip_model, clip_tokenizer, n_words=None, ixtoword=None, clip_transform=None):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        #torch.cuda.set_device(cfg.GPU_ID)
        #cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.dataset = dataset
        self.num_batches = len(self.data_loader)
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.clip_transform = clip_transform

    def build_models(self):

        def count_parameters(model):
            total_param = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_param = np.prod(param.size())
                    if param.dim() > 1:
                        print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                    else:
                        print(name, ':', num_param)
                    total_param += num_param
            return total_param

        # ###################encoders######################################## #
        # initialize CLIP with more configs here


        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:

        # print('number of trainable parameters =', count_parameters(netG))
        # print('number of trainable parameters =', count_parameters(netsD[-1]))

        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(filter(lambda p: p.requires_grad, netsD[i].parameters()),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, real_image, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'% (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        # att_sze = region_features.size(2)
        att_sze = math.sqrt((region_features.size(2) - 1)) # 50 => 49 => 7 == att_sze
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
        print(real_image.type)


    def train(self):
        netG, netsD, start_epoch = self.build_models()
        clip_model = self.clip_model
        clip_transform = self.clip_transform

        device_ids = [i for i in range(torch.cuda.device_count())]
        print("device_ids", device_ids)
 
        netG = nn.DataParallel(netG, device_ids)
        for i in range(len(netsD)):
            netsD[i] = nn.DataParallel(netsD[i], device_ids)

        clip_model.backbone.text_model = nn.DataParallel(clip_model.backbone.text_model, device_ids)
        clip_model.backbone.vision_model = nn.DataParallel(clip_model.backbone.vision_model, device_ids)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        real_labels_2, fake_labels_2, match_labels_2 = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        print("cfg", cfg)
        print("CFGCUDA", cfg.CUDA)
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        
        mask = mask_correlated_samples(self)
        print(mask.shape)

        temperature = 0.5
        device = noise.get_device()
        print("device", device)
        mask = mask.to(device)
        criterion = NT_Xent(batch_size, temperature, mask, device)
        
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                
                # imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                # imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
                # sort_ind, sort_ind_2 = prepare_data(data, self.clip_tokenizer)
                

                imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
                sort_ind, sort_ind_2 = data
                # attention mask
                mask = captions['attention_mask']
                mask_2 = captions_2['attention_mask']

                # if cfg.CUDA:
                #     captions = captions.cuda()
                #     captions_2 = captions_2.cuda()
                

                ##hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                ## words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                ## words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                # print("captions", captions)
                # words_embs, sent_emb = clip_encode_text(clip_model, captions)
                words_embs, sent_emb = clip_model.encode_text_verbose(**captions)
                words_embs_2, sent_emb_2 = clip_model.encode_text_verbose(**captions_2)
                
                #######################################################
                # (2) Generate fake images
                ######################################################
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                #mask = (captions == 0)
                #num_words = words_embs.size(2)
                #if mask.size(1) > num_words:
                #    mask = mask[:, :num_words]


                words_embs_2, sent_emb_2 = words_embs_2.detach(), sent_emb_2.detach()
                #mask_2 = (captions == 0)
                #num_words = words_embs_2.size(2)
                #if mask.size(1) > num_words:
                #    mask_2 = mask_2[:, :num_words]


                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask, cap_lens)
                fake_imgs_2, _, mu_2, logvar_2 = netG(noise, sent_emb_2, words_embs_2, mask_2, cap_lens_2)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD, log = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    errD_2, log_2 = discriminator_loss(netsD[i], imgs_2[i], fake_imgs_2[i],
                                                sent_emb_2, real_labels_2, fake_labels_2)
                    
                    errD += errD_2
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())
                    D_logs += log
                    
                    D_logs += 'errD%d_2: %.2f ' % (i, errD_2.item())  + log_2


                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs, cnn_code = \
                    generator_loss(netsD, clip_model, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids, clip_transform=clip_transform)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                
                
                errG_total_2, G_logs_2, cnn_code_2 = \
                    generator_loss(netsD, clip_model, fake_imgs_2, real_labels_2,
                                   words_embs_2, sent_emb_2, match_labels_2, cap_lens_2, class_ids_2, clip_transform=clip_transform)
                kl_loss_2 = KL_loss(mu_2, logvar_2)
                errG_total_2 += kl_loss_2
                G_logs_2 += 'kl_loss: %.2f ' % kl_loss_2.item()

                G_logs += G_logs_2

                errG_total += errG_total_2


                _, ori_indices = torch.sort(sort_ind, 0)
                _, ori_indices_2 = torch.sort(sort_ind_2, 0)

                # total_contra_loss = 0
                # i = -1
                cnn_code = cnn_code[ori_indices]
                cnn_code_2 = cnn_code_2[ori_indices_2]

                cnn_code = l2norm(cnn_code, dim=1)
                cnn_code_2 = l2norm(cnn_code_2, dim=1)


                contrastive_loss = criterion(cnn_code, cnn_code_2)
            

                # contrative_loss = contrative_loss * 0.2
                # print("contrative_loss", contrative_loss)
                contrastive_loss = contrastive_loss * 0.2

                G_logs += 'contrastive_loss: %.2f ' % contrastive_loss.item()

                errG_total += contrastive_loss
                
                
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print('Epoch [{}/{}] Step [{}/{}]'.format(epoch, self.max_epoch, step,
                                                              self.num_batches) + ' ' + D_logs + ' ' + G_logs)
                # # save images
                if gen_iterations % 10000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    # self.save_img_results(netG, fixed_noise, sent_emb, words_embs, mask, image_encoder,
                    #                      captions, cap_lens, epoch, imgs[-1], name='average')
                    load_params(netG, backup_para)
                    
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            print('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
                epoch, self.max_epoch, errD_total.item(), errG_total.item(), end_t - start_t))
            print('-' * 89)
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)
                torch.save(clip_model.state_dict(),
                           '%s/clip%d.pth' % (self.model_dir, epoch))

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        clip_model = self.clip_model

        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            R = np.zeros(30000)
            cont = True
            for ii in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                if (cont == False):
                    break
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if (cont == False):
                        break
                    if step % 100 == 0:
                       print('cnt: ', cnt)

                    imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
                sort_ind, sort_ind_2 = prepare_data(data, self.clip_tokenizer)

                    words_embs, sent_emb = clip_model.encode_text_verbose(captions)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            #print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                        im.save(fullpath)

                    # _, cnn_code = image_encoder(fake_imgs[-1])

                    # fake_im = Image.fromarray(fake_imgs[-1])
                    # fake_im.save("FAKE.png")
                    clip_resized = F.interpolate(fake_imgs[-1], size=clip_model.backbone.vision_model.config.image_size)
                    # clip_resized = F.interpolate(fake_imgs[], size=clip_model.visual.input_resolution)
                    # clip_resized_im = Image.fromarray(clip_resized)
                    # clip_resized_im.save("RESIZED.png")

                    _, cnn_code = clip_model.encode_image_verbose(clip_resized)
                    
                    
                    for i in range(batch_size):
                        mis_captions, mis_captions_len = self.dataset.get_mis_caption(class_ids[i])

                        _, sent_emb_t = clip_model.encode_text_verbose(mis_captions)
                        rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                        ### cnn_code = 1 * nef
                        ### rnn_code = 100 * nef
                        scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                        cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                        rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                        norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                        scores0 = scores / norm.clamp(min=1e-8)
                        if torch.argmax(scores0) == 0:
                            R[R_count] = 1
                        R_count += 1

                    if R_count >= 30000:
                        sum = np.zeros(10)
                        np.random.shuffle(R)
                        for i in range(10):
                            sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                        R_mean = np.average(sum)
                        R_std = np.std(sum)
                        print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                        cont = False





    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
