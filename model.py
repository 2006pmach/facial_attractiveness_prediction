import torch
import torch.nn as nn
from torchvision import models


def build_model(arch, num_classes, pretrained=True):
    model = models.__dict__[arch](pretrained=pretrained)

    model = change_linear_layer(model, arch, num_classes)

    return model


def change_linear_layer(model, arch, num_classes):
    if "densenet" in arch:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif ("mobilenet" in arch) or ("mnasnet" in arch) or ("alexnet" in arch) or ("vgg" in arch):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif "squeezenet" in arch:
        in_channels = model.classifier[1].in_channels
        kernel_size = model.classifier[1].kernel_size
        stride = model.classifier[1].stride
        model.classifier = nn.Conv2d(in_channels, num_classes, kernel_size, stride)
    else:
        # googlenet, inception_v3 only without avgpool, resnet, resnext, shufflenet, wide_resnet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def build_criterion(num_classes, weight=None):
    if num_classes == 1:
        criterion = nn.MSELoss(reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean")

    return criterion
