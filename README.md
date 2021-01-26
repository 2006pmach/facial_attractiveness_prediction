# Hotness Prediction using Facial Images

This repository provide the code to assess the attraciveness of medical images.
The models are trained on a private dataset obtained from the BLINQ dating app and further fine-tuned using the [Chicago Face Dataset](https://chicagofaces.org/default/). We provide pretrained and the fine-tuned models in order to predict the attractiveness given a facial image.


## Requirements
```
Python 3
pytorch>=1.3
torchvision
pillow
```
## Setup

Please download the CNN pytorch model weight including the folders [here](https://drive.google.com/drive/folders/1T3ixZq2LFCq-tV2MOv2TD8ElHlmPR2qR?usp=sharing) and copy them into the _checkpoints_ folder. 

## Running the Scripts

`python predict_hotness.py predict_bins_hotness_densenet201_cfd.py images/merkel.jpg`

or

`python predict_hotness.py predict_bins_hotness_densenet201_cfd.py images/merkel.jpg`

## Examples

We run our model for demonstration purposes here on two publicly available images from the Web.
