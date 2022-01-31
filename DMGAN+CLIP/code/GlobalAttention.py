"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""
import math
import torch
import torch.nn as nn


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1, query_mask):
    """Calculate (word, patch_region) attn scores and patch_region embedding vectors per word.
    If there are ten images and each image has 49 patches (source), then we have 490 patches.
    If there are 30 words in each sentence (query),
    We're calculating a 30 * 490 attention scores.
    
    Parameters:
        query: batch x emb_size x n_words
        context: batch x emb_size x n_patches
        query_mask: batch_size x 1 x queryL
    Returns:
        weightedContext: batch x n_words x emb_size
        attn: batch x n_words x 7 x 7
    """
    batch_size, emb_size, n_words = query.shape
    n_patches = context.size(2)
    att_sze = int(math.sqrt(n_patches)) # 50 => int(sqrt(50)) => 7 == att_sze because CLIP has {[CLS], ...7x7 patches}

        
    contextT = torch.transpose(context, 1, 2).contiguous()     
    queryT = torch.transpose(query, 1, 2).contiguous()

    contextT = l2norm(contextT, dim=2)
    queryT = l2norm(queryT, dim=2)
    query = torch.transpose(queryT, 1, 2)
    # for img, text in zip(contextT, queryT):
    #     for patch in zip(img, text):
    #         print(patch[0])
    #         print(patch[1])
    #         print(patch[0].unsqueeze(-1) * torch.transpose(patch[1].unsqueeze(0), 0, 1))
    #         print()

    query_mask = query_mask.repeat(1, n_patches, 1) 
    assert query_mask.shape == (batch_size, n_patches, n_words) # (10 images, 49 patches, 1 for valid token, 0 for padding token.
    
    '''
    added query_masks because we shouldn't be calculating and using attention scores for
    padded tokens. As a reference the following is from huggingface BERT code:

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    And this is ours:

        attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
        attn = attn * query_mask
        attn = nn.functional.softmax(attn, dim=-1)  # Eq. (8)  
    '''

    assert contextT.shape   == (batch_size, n_patches, emb_size)
    assert query.shape      == (batch_size, emb_size, n_words)      
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper. 
    assert attn.shape == (batch_size, n_patches, n_words)  # Attn Logit for (patch_i, word_j) for all i,j comboination, for all images

    '''

    tensor([ -56.4774,   40.9221,   -2.1741,  -57.5465,   -6.6706,  -65.0118,
            -81.4406,  -44.2379,  -86.3934,  -20.4601,  -73.8708,  -67.5609,
            -80.7010,  -90.2514,  -72.4438,  -73.2443,  -92.2100, -105.7497,
            -91.4868,  -68.0369,  -69.2500,  137.6093,  -50.6344,  -52.6845,
            -53.4920,  -53.6696,  -53.6282,  -53.7165,  -53.7873,  -53.5998],
        device='cuda:0', grad_fn=<UnbindBackward>)
    '''
    
    attn = attn.masked_fill_(query_mask == 0, -float("inf")) # mask out padding tokens
    attn = nn.functional.softmax(attn, dim=-1) # attn scores from [0.0, 1.0] for each (patch, word) pair
    # for img in attn[:1]:
    #     for patch in img[10:14]:
    #         print(patch)
    #         print()
    # '''
    # >>> tensor([0.0622, 0.0740, 0.0755, 0.0608, 0.0553, 0.0584, 0.0619, 0.0635, 0.0564,
    #     0.0590, 0.0591, 0.0567, 0.0549, 0.0531, 0.0543, 0.0950, 0.0000, 0.0000,
    #     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #     0.0000, 0.0000, 0.0000], device='cuda:0', grad_fn=<UnbindBackward>)
    # '''
        
    '''
    The above code finishes to equation 8 in the AttnGAN paper, where we compute the attention vector across all the words
    for each patch in context. The output of equation 8 is the "attn" variable.

    The code below is for equation 9, where we're calculating an region-vector for each word.
    I changed attn to attn2 to separate the two.

    With the original code, it'd be suggesting that the output of equation 8 is:
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)
    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)

    which is crazy!
    '''


    attn2 = torch.transpose(attn, 1, 2).contiguous()
    assert attn2.shape == (batch_size, n_words, n_patches)
    attn2 = attn2.view(batch_size * n_words, n_patches)

    '''
    Eq. (9)
    Then, we build an attention model to compute a region context vector for each word (query). The region-context
    vector ci is a dynamic representation of the image's subregions related to the ith word 
    of the sentence. It is computed as the weighted sum over all regional visual vectors,
    '''
    attn2 = attn2 * gamma1
    attn2 = nn.functional.softmax(attn2, dim=-1)
    attn2 = attn2.view(batch_size, n_words, n_patches) 
    attnT = torch.transpose(attn2, 1, 2).contiguous()

    assert context.shape == (batch_size, emb_size, n_patches)
    assert attnT.shape == (batch_size, n_patches, n_words)
    weightedContext = torch.bmm(context, attnT)

    assert weightedContext.shape == (batch_size, emb_size, n_words)
    attn = attn.view(batch_size, att_sze, att_sze, n_words)
    # print(attn[0,:2, :2])
    attn = attn.permute(0, 3, 1, 2)
    weightedContext = weightedContext.permute(0,2,1)
    return weightedContext, attn


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        #self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context_key, content_value):#
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context_key.size(0), context_key.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        #sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        #sourceT = self.conv_context(sourceT).squeeze(3)
        sourceT = context_key

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)

        text_weighted = None
        # text_attn = torch.transpose(attn, 1, 2).contiguous() # batch x sourceL x queryL
        # text_attn = text_attn.view(batch_size*sourceL, queryL)
        # if self.mask is not None:
        #     mask = self.mask.repeat(queryL, 1)
        #     mask = mask.view(batch_size, queryL, sourceL)
        #     mask = torch.transpose(mask, 1, 2).contiguous()
        #     mask = mask.view(batch_size*sourceL, queryL)
        #     text_attn.data.masked_fill_(mask.data, -float('inf'))
        # text_attn = self.sm(text_attn)
        # text_attn = text_attn.view(batch_size,sourceL, queryL)
        # text_attn = torch.transpose(text_attn, 1, 2).contiguous() # batch x queryL x sourceL
        # # (batch x idf x queryL) * (batch x queryL x sourceL) -> batch x idf x sourceL
        # text_weighted = torch.bmm(target, text_attn)

        # --> batch*queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(content_value, attn)  #
        #weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


class GlobalAttention_text(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttention_text, self).__init__()
        self.conv_context = nn.Conv1d(cdf, idf, kernel_size=1, stride=1, padding=0)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = self.conv_context(context)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)

        # --> batch*queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        #attn_o = self.sm(attn)  # Eq. (2)
        #attn_o = attn_o.view(batch_size, queryL, sourceL)

        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.nn.Softmax(dim=1)(attn)

        #import ipdb;
        #ipdb.set_trace()  # BREAKPOINT

        # (batch x idf x queryL) * (batch x queryL x sourceL) -> batch x idf x sourceL
        text_weighted = torch.bmm(target, attn)

        return text_weighted

