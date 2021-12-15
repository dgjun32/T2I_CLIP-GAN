import torch
import torch.nn as nn
from PIL import Image

import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# import torchvision.transforms.functional.normalize as TF

def forward(clip_model, image, text):
    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = clip_model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits_per_image, logits_per_text = clip_model(image, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    # shape = [global_batch_size, global_batch_size]
    return probs


# ##################Loss for DAMSM training###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        # masks = torch.ByteTensor(masks)
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 7 x 7
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 49
        context = img_features.reshape(batch_size, 512, 49)
        """
            word(query): batch x nef x words_num
            context: batch x nef x 7*7
            weiContext: batch x nef x words_num
            attn: batch x words_num x 7 x 7
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        # masks = torch.ByteTensor(masks)
        masks = torch.BoolTensor(masks)

        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.module.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.module.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.module.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.module.UNCOND_DNET is not None:
        real_logits = netD.module.UNCOND_DNET(real_features)
        fake_logits = netD.module.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    log = 'Real_Acc: {:.4f} Fake_Acc: {:.4f} '.format(torch.mean(real_logits).item(), torch.mean(fake_logits).item())
    return errD, log


def generator_loss(netsD, clip_model, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids, clip_transform):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    image_encodings = []
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].module.COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].module.UNCOND_DNET is  not None:
            logits = netsD[i].module.UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.data[0]
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef

              # # testing interpolation.
            # im = fake_imgs[i][0].detach().cpu().numpy()
            # im = (im + 1.0) * 127.5
            # im = im.astype(np.uint8)
            # im = np.transpose(im, (1, 2, 0))
            # im = Image.fromarray(im)
            # im.save("FAKE.png")
            # clip_resized_im = F.interpolate(fake_imgs[i], size=clip_model.backbone.vision_model.config.image_size)
            # im = clip_resized_im[0].detach().cpu().numpy()
            # im = (im + 1.0) * 127.5
            # im = im.astype(np.uint8)
            # im = np.transpose(im, (1, 2, 0))
            # im = Image.fromarray(im)
            # im.save("RESIZED.png")
            # raise Exception()
            clip_resized = F.interpolate(fake_imgs[i], size=clip_model.backbone.vision_model.module.config.image_size)
            region_features, image_encoding = clip_model.encode_image_verbose(clip_resized)
            region_features = region_features[:,:,1:].reshape(-1, 512, 7, 7)
    
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            s_loss0, s_loss1 = sent_loss(
                image_encoding, sent_emb, match_labels, class_ids, batch_size
            )
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            image_encodings.append(image_encoding)
            
            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())

    return errG_total, logs, image_encoding


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
