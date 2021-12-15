from __future__ import print_function

from miscc.config import cfg, cfg_from_file
# from datasets import TextDataset
from datasets import TextDataset, prepare_data
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
import transformers
from torch.utils.data._utils.collate import default_collate
from transformers import CLIPModel, CLIPTokenizer

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


from model import AddLinearOnCLIP 


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DMGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/clip_bird_DMGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--NET_G', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().decode('utf8').split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().decode('utf8').split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.NET_G != '':
        cfg.TRAIN.NET_G = args.NET_G

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
    torch.cuda.set_device(cfg.GPU_ID)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print("Seed: %d" % (args.manualSeed))

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    clip_model = AddLinearOnCLIP()

    if os.path.exists(cfg.TRAIN.CLIP_MODEL_CHECKPOINT):
        # load pretrained CLIP
        clip_model.load_state_dict(torch.load(cfg.TRAIN.CLIP_MODEL_CHECKPOINT, map_location="cpu"))
        print(cfg.TRAIN.CLIP_MODEL_CHECKPOINT)
    else: 
        clip_model.backbone = CLIPModel.from_pretrained(cfg.TRAIN.CLIP_MODEL_BASE)

    clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.TRAIN.CLIP_MODEL_BASE)

    if cfg.CUDA:
        clip_model = clip_model.cuda()

    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
        
    dataset = TextDataset(
        cfg.DATA_DIR, 
        split_dir,
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform,
    )
    assert dataset

    def collate_fn(batch):
        batch = default_collate(batch)
        return prepare_data(batch, clip_tokenizer)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS), collate_fn=collate_fn)

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset, clip_model, clip_tokenizer, clip_transform=None) 

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
