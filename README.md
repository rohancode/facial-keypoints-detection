# face-keypoint

## Introduction

Real time face keypoint detection on mobile gpus. Face keypoint detection is used in many applications like applying makeup, 
instagram filters etc.Generally face keypoint or body keypoint detection are solved using hourglass or 
encoder-decoder CNN architectures, but those architectures large and slow for edge cases.
So I am trying to solve this problem by treating it as a regression problem on an open source dataset 
i.e. directing predicting x and y coordinates of the keypoint by using MobilenetV1 (pretrained on ImageNet).

When building deep learning models for mobile device especially for real-time use cases (fps >= 30 ) inference and
model size are the most important things to keep in mind.

There are a few ways to increase inference time:
   * Decrease input image size.
   * Use smaller networks like MobileNets.
   * If making custom models try to use as less layers as possible.
      Decreasing depth of the model increase the inference more the decreasing the width of the model.

## Quick Start
  1. Download the dataset from [here](https://wywu.github.io/projects/LAB/LAB.html) and extract in `data` folder.
  2. Run `convert_data.py` to process the data.
      It processes the given label formet and extract faces from data images and 
      saves into `train_images`  and `test_images` inside `data` folder.
      ```
      python convert_data.py
      ```
  3. Run `train.py` to train model and save weights in `weights`.
      ```
      python train.py
      ```
  4. Run 'convert_model.py' to convert saved h5 model to tflite.
      ```
      python convert_model.py --model_filename face_keypoint_mobile.h5 --tflite_filename face_keypoint_mobile.tflite
      ```
     
## Results

<p align="center">
  <img src='media/prediction.png' width="1000" height="400">
</p>

<p align="center">
  <em>Predictionexmaples examples</em>
</p>

<p align="center">
  <img src='media/label.png' width="1000" height="400">
</p>

<p align="center">
  <em>Groud truth examples</em>
</p>
