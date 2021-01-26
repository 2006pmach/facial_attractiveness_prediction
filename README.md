# Hotness Prediction using Medical Facial Images

This repository provide the code to assess the attraciveness of medical images.
The models are: BLINQ- trained on a private dataset obtained from the BLINQ dating app and MEDICAL- the further fine-tuned model using the [Chicago Face Dataset](https://chicagofaces.org/default/).
We provide the pretrained and the fine-tuned models in order to predict the attractiveness given a facial image.

The maximal score is 10 and the minimal 1. However, in BLINQ the assigned scores tend to be lower than on CFD such that 10s are very rare in the former.


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

`python predict_hotness.py predict_bins_hotness_densenet201_blinq images/merkel.jpg`

or

`python predict_hotness.py predict_bins_hotness_densenet201_medical images/merkel.jpg`

## Examples

We run our model for demonstration purposes here on two publicly available images from the Web (downloaded from Wikipedia).

![alt text](https://github.com/2006pmach/facial_attractiveness_prediction/blob/main/images/Angela_Merkel_wikipedia.jpg "Angela Merkel")

This picture of Angela Merkel achieves an attractiveness score of 

`2.829862910516840` with a standard deviation of `0.0877247232573163` (BLINQ) and 

`3.557292120774966` with a standard deviation of `0.2021892417092418` (Medical). 

Note, that we feed the image 20 times through the CNN with random augmentations to compute the mean and standard deviation.

![alt text](https://github.com/2006pmach/facial_attractiveness_prediction/blob/main/images/Katheryn_Winnick_wikipedia.jpg "Katheryn Winnick")

This picture of Katheryn Winnick achieves an attractiveness score of 

`6.378340877147275` with a standard deviation of `0.13951626876974346` (BLINQ) and 

`7.932078346675053` with a standard deviation of `0.03819095674717804` (Medical). 
