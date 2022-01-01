import os
import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import torchvision
from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms
import PIL
from PIL import Image

import numpy as np
from scipy.stats import entropy

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def inception_score(path, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    path -- path containing fake images
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    files = []
    mid_files = os.listdir(path)
    for dir in mid_files:
        files_li = os.listdir(os.path.join(path, dir))
        files_li = list(map(lambda x : os.path.join(path, dir, x), files_li))
        files += files_li
    print(files[30])
    N = len(files)
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    transform = torchvision.transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = ImagePathDataset(files, transforms = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    # dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str,
                    help='dataset name')
    args = parser.parse_args()
    
    if args.data == 'bird':
        path = '../models/netG_bird/valid/single/'
    else:
        path = '../models/netG_coco/valid/single/'
    print ("Calculating Inception Score...")
    mu, std = inception_score(path, cuda=True, batch_size=32, resize=True, splits=10)
    print ('scores mean : {} | scores std : {}'.format(mu, std))