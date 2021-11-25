import os
import argparse
import importlib
import numpy as np
import pandas as pd
from PIL import Image
from shutil import copyfile, copy2

import torch
import torch.nn as nn
from torchvision import models, transforms


def load_checkpoint(checkpoint_dir, model_name="checkpoint.pth", cuda_is_available=True, seed=0):
    model_path = os.path.join(checkpoint_dir, str(seed))

    if seed == 0 and not os.path.exists(model_path):
        model_path = checkpoint_dir

    print(model_path)

    if cuda_is_available:
        checkpoint = torch.load(os.path.join(model_path, model_name))
    else:
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cpu'))

    return checkpoint


def build_pred_transformations(params, convert_gray_to_rgb=False):
    transform_list = []

    if convert_gray_to_rgb:
        transform_list.append(transforms.Lambda(lambda img: img.convert('RGB')))

    transform_list.append(transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), resample=Image.BICUBIC))
    transform_list.append(transforms.Resize(params.img_shape))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)

