# Finetuning Pretrained CLIP using DAMSM and Constrastive Loss for text to image synthesis

## 1. Methodology
Neural network for Text to Image generation is composed of 2 sub-networks. 

```Text Encoder``` and ```Generator Network``` 

Therefore, It requires two-step training to train text-to-image generator.

1. Image Encoder and Text Encoder are jointly pretrained from image-caption pair thereby projecting image and text to common space.
2. After text encoder pretraining, Generator Network is advarsarialy trained to generate realistic image based on text feature.

Recent research proposed using <b>DAMSM loss + Contrastive loss</b> for pretraining text encoder and training ```DM-GAN```, thereby reaching SOTA.

In this work, We replaced RNN based text encoder and CNN based image encoder with ```CLIP```, which is pretrained multimodal ```Vision Language Model``` based on transformer architecture.  

## 2. CLIP

```CLIP``` is multimodal encoder for image and natural language, which is pretrained using ```contrastive loss``` with huge batch size(=32768). 

This is link for paper and official pytorch implementation of [CLIP](https://openai.com/blog/clip/)

## 3. Prepared Data

Download the preprocessed datasets from [AttnGAN](https://github.com/taoxugit/AttnGAN)

Alternatively, another site is from [DM-GAN](https://github.com/MinfengZhu/DM-GAN)

## 4. Trained model


## 5. Training
1. Fine tuning pretrained CLIP encoder

- With CUBS2011 using DAMSM + Contrastive loss : ```$ python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0```

- With COCO2014 using DAMSM + Contrastive loss : ```$ python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 0```

2. Training DM-GAN

- With CUBS2011 : ```$ python main.py --cfg cfg/clip_bird_DMGAN.yml --gpu 0```

- With COCO2014 : ```$ python main.py --cfg cfg/clip_coco_DMGAN.yml --gpu 0```

## 6. Evaluation
1. Generate fake images and compute ```R precision```

- CUBS2011 : ```$ python main.py --cfg cfg/eval_clip_bird.yml```

- COCO2014 : ```$ python main.py --cfg cfg/eval_clip_coco.yml```

2. Compute ```FID(Frechet Inception Distance)``` 

- CUBS2011 : ```$ python fid_score.py --path ./CLIP+GAN/DMGAN+CLIP/output/netG_bird/valid/single/ ./CLIP+GAN/data/birds/test/ --dims 2048 --batch_size 32```

- COCO2014 : ```$ python fid_score.py --path ./CLIP+GAN/DMGAN+CLIP/output/netG_coco/valid/single/ ./CLIP+GAN/data/coco/val2014/ --dims 2048 --batch_size 32```
 
3. Compute ```Inception score``` : 

## 7. Citation

