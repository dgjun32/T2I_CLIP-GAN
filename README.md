# Finetuning Pretrained CLIP using DAMSM and Constrastive Loss for text to image synthesis

## 1. Methodology
Neural network for Text to Image generation is composed of 2 sub-networks. 
```Text Encoder``` and ```Generator Network``` 

Image Encoder and Text Encoder are jointly pretrained from image-caption pair thereby fusing image and text information. After pre-training text encoder, GAN is advarsarially trained to generate realistic image based on text feature.

Recent research proposed using <b>DAMSM loss + Contrastive loss</b> for pretraining text encoder, thereby reaching SOTA.

In this work, We replaced RNN based text encoder with ```CLIP```, which is pretrained multimodal(image-language) encoder based on transformer architecture.  

## 2. CLIP

CLIP is multimodal encoder for image and natural language, which is pretrained by ```contrastive loss``` with very large batch size(32768).

## 3. Prepared Data

## 4. Trained model

## 5. Evaluation

## 6. Citation

