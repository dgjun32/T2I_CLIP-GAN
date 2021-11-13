# Finetuning Pretrained CLIP using DAMSM and Constrastive Loss for text to image synthesis

## 1. Methodology
Neural network for Text to Image generation is composed of 2 sub-networks. 

```Text Encoder``` and ```Generator Network``` 

Image Encoder and Text Encoder are jointly pretrained from image-caption pair thereby projecting image and text to common space. After text encoder pretraining, Generator Network is advarsarialy trained to generate realistic image based on text feature.

Recent research proposed using <b>DAMSM loss + Contrastive loss</b> for pretraining text encoder, thereby reaching SOTA.

In this work, We replaced RNN based text encoder and CNN based image encoder with ```CLIP```, which is pretrained multimodal ```Vision Language Model``` based on transformer architecture.  

## 2. CLIP

```CLIP``` is multimodal encoder for image and natural language, which is pretrained using ```contrastive loss``` with very large batch size(=32768). 

This is link for paper and official pytorch implementation of [CLIP](https://openai.com/blog/clip/)

## 3. Prepared Data

Download the preprocessed datasets from [AttnGAN](https://github.com/taoxugit/AttnGAN)

Alternatively, another site is from [DM-GAN](https://github.com/MinfengZhu/DM-GAN)

## 4. Trained model


## 5. Training
- Finetune pretrained CLIP with CUBS2011 using DAMSM + CL : ```python pretrain_DAMSM_huggingface.py --cfg cfg/DAMSM/bird.yml --gpu 1```

- Train DM-GAN with CUBS2011 : 

## 6. Evaluation


## 7. Citation

