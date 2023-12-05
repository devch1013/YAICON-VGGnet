<img src="https://github.com/devch1013/YAICON-VGGnet/assets/92439610/0ce35aae-3e19-4a7c-9ea2-0fd8434a6edb" width = "900" >

# VGGnet: Video Graphic Generation network

## 3rd YAICON Novelty Prize!!


## Members 
- 박찬혁: PM, AI lead
- 최가윤: AI
- 박승호: AI 
- 유선재: Data
- 제갈건: Data
  
---
</br>
</br>

**This is an attempt to combine the video generation model with ImageBind to create a pipeline that can generate video even with multimodal input.**

## 1. Dataset
We prepare text-text, image-text, audio-text dataset to generate embedding pair of ImageBind and T5 embedding model.


## 2. Mapping Model

## Train
```
python -m embedding.mapper_train
```

## Inference
Change the promts to text or image path or audio path.
```
python -m run_inference
```








