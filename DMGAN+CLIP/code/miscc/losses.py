import torch
import torch.nn as nn
from PIL import Image

import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# import torchvision.transforms.functional.normalize as TF
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


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


# text-image similarity module(for computing word_loss)
def similarity_text_image(words_emb, region_features, words_mask, gamma1, gamma2):
    """
    Calculates the 1-1 similarity between words_emb and region_features.
    
    words_emb(query): batch_size x emb_size x n_words(30-1) 
    words_mask: batch_size x n_words(30-1) x 1
    region_features(context): batch_size x emb_size x n_patches(49)
    """
    batch_size, emb_size, n_words = words_emb.size()
    batch_size, emb_size, n_patches = region_features.size()
    assert words_mask.size() == (batch_size, n_words, 1)

    
    '''FROM PAPER
    We first calculate the similarity matrix for all possible
    pairs of words in the sentence and sub-regions in the image 
    29 words, 49 patches for a total of 49 * 29 similarity scores.
    '''
    contextT = torch.transpose(region_features, 1, 2).contiguous() # shape of (batch_size, 49, 512)     
    queryT = torch.transpose(words_emb, 1, 2).contiguous() # shape of (batch_size, 29, 512)
    contextT = l2norm(contextT, dim=2)
    queryT = l2norm(queryT, dim=2)
    sim_scores = torch.bmm(queryT, torch.transpose(contextT, 1, 2)) # shape of (batch_size, 29, 49)
    assert sim_scores.size() == (batch_size, n_words, n_patches)
    '''NOT FROM PAPER
    We should be masking out the similarity scores corresponding to padding tokens.
    
        sim_scores.shape == [batch_size, n_words, n_patches]
        words_mask.shape == [batch_size, n_words, 1]
    '''
    

    sim_scores = sim_scores.masked_fill_((words_mask == 0).cuda(), -float("inf")) # mask out padding tokens
    
    '''
    TEST: 1. The similarity scores between ith word and jth patch is set to -inf if
    ith word is a padding
    '''
    one_mask = words_mask[0]
    one_scores = sim_scores[0] 
    for m, s in zip(one_mask, one_scores):
        if m == 0.0:
            assert s[0] == -float("inf")
    
    '''FROM PAPER
    We find that it is beneficial to normalize the
    similarity matrix as follows (softmax)
    '''
    sim_scores = torch.transpose(sim_scores, 1, 2) # shape of (batch_size, 49, 29)
    sm_sim_scores = nn.functional.softmax(sim_scores, dim=-1)
    assert torch.isclose(torch.sum(sm_sim_scores[0][0]), torch.tensor([1.0]).to(sm_sim_scores.get_device()), rtol=1e-5) 
    
    '''
    sim_scores.shape == [batch_size, n_patches, n_words]
    '''
    
    '''
    TEST 2. The similarity scores between the ith word and the kth patch is set to 0 if it's padding.
    sim_scores.shape == [batch_size, n_patches, n_words], so let's just consider one patch.
    '''
    one_mask = words_mask[0]
    one_scores = sm_sim_scores[0][0] # one patch
    for m, s in zip(one_mask, one_scores):
        if m == 0.0:
            assert s == 0.0

    '''FROM PAPER
    Then, we build an attention model to compute a region-context vector for each word (query). 
    The region-context vector ci is a dynamic representation of the image's subregions related to the 
    ith word of the sentence. It is computed as the weighted sum over all regional visual vectors.

    ci = sum(alpha_ij * v_j) from j = 0 to j = 49 where 
        alpha_j = sim score of ith word with jth patch
        v_j = embedding for jth patch
    '''

    # sm_sim_scores.size() == [10, 49, 29]
    # contextT.size() == [10, 49, 512]
    scores_for_rc_vector = gamma1 * sm_sim_scores
    scores_for_rc_vector = nn.functional.softmax(scores_for_rc_vector, dim=1)
    assert torch.isnan(scores_for_rc_vector).sum() == 0
    scores_for_rc_vector = scores_for_rc_vector.permute(0, 2, 1) # [10, 29, 49]
    '''
    scores_for_rc_vector.shape == [batch_size, n_words, n_patches]
    '''

    # region_context_rc_vector.size() == [10, 29, 512]
    region_context_vectors = torch.bmm(scores_for_rc_vector, contextT)
    assert torch.isnan(region_context_vectors).sum() == 0
    '''FROM PAPER
    Finally, we define the relevance between the 
    ith word and the image using the cosine similarity between ci and ei.

    R(ci, ei) = cossim(ci, ei)

    Inspired by the minimum classification error formulation in speech recognitioregion_context_vectorsn
    (see, e.g., [11, 8]), the attention-driven image-text matching score between the 
    entire image (Q) and the whole text description (D) is defined as

    [refer to paper]
    R(Q, D) = ...
    '''
    cossim = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
    R_ci_ei = cossim(region_context_vectors, queryT) # shape of (batch_size*n_words, batch_size*n_words)
    R_QD = R_ci_ei * gamma2
    R_QD = R_QD.exp_()
    R_QD = R_QD.sum(dim=1)
    R_QD = torch.pow(R_QD, (1 / gamma2))
    R_QD = torch.log(R_QD)
    
    '''
    In Summary,
    
    sm_sim_scores                ([batch_size, n_patches, n_words])
        for each patch, the attn scores across the full sequence
    region_context_vectors       ([batch_size, n_words, emb_size])
        for each word, the weighted sum of all the patch embeddings
        i.e. a full image embedding corresponding to each "word"
    R_QD                         ([batch_size])
        Similarity score for ith text with ith image for all i in range(batch_size)
    '''
    return sm_sim_scores, region_context_vectors,  R_QD


def words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size, words_mask, gamma1, gamma2, gamma3):
    """
        words_emb(query): batch_size x emb_size x n_words
        region_features(context): batch_size x emb_size x n_patches
    """
    masks = []
    attn_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            classmask = (class_ids == class_ids[i]).astype(np.uint8)
            classmask[i] = 0
            masks.append(classmask.reshape((1, -1)))
        # Get the i-th text description
        word = words_embs[i].unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)
        word_mask = words_mask[i].contiguous()
        word_mask = word_mask.repeat(batch_size, 1)
        word_mask = word_mask.unsqueeze(-1)
        context = region_features
        
        # attn, rc_vectors, R_QD between i th text and all images
        attn, rc_vectors, R_QD = similarity_text_image(
            word, 
            context, 
            word_mask, 
            gamma1, 
            gamma2
        )
        attn_maps.append(attn)
        # sim of all the images in the batch against one text description
        similarities.append(R_QD)
    
    # similarities[i][j] = similarity(text_i, image_j)
    similarities = torch.stack(similarities) * gamma3

    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()
    # if there are overlapping labels within batch
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
        
    similarities1 = similarities.transpose(0, 1)

    if match_labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, match_labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, match_labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, attn_maps


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
