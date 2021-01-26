import os
import torch
import argparse
import importlib
import numpy as np
from PIL import Image

import utils
from model import build_model, build_criterion


def get_params(params_name=None, input_img_path=None):
    if params_name is None and input_img_path is None:
        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        parser.add_argument("params", type=str, help="Name of parameter file in parameters package.")
        parser.add_argument("input_img_path", type=str, help="Path to input image to predict hotness.")
        args = parser.parse_args()
        params_name = args.params
        input_img_path = args.input_img_path

    params_module = importlib.import_module('parameters.{}'.format(params_name))

    params_class = params_module.Parameters
    params = params_class()

    params.input_img_path = input_img_path

    return params


def main(params):
    predict_transform = utils.build_train_transformations(params)

    img = Image.open(params.input_img_path)

    img_aug = torch.stack([predict_transform(img) for _ in range(0,20)], dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format("GPU" if torch.cuda.is_available() else "CPU"))

    model = build_model(params.arch, params.num_classes)
    model = model.to(device)

    ckpts = utils.load_checkpoint(params.model_path, model_name="best_model.pth")
    model.load_state_dict(ckpts["state_dict"])

    model.eval()

    with torch.no_grad():
        p_hotness = model(img_aug.to(device))
        prob_hotness = torch.softmax(p_hotness, dim=1)
        prob_hotness = prob_hotness.cpu().detach().numpy()

        if params.num_classes > 1:
            pred_cont = prob_hotness.dot(np.arange(0, params.num_classes).reshape(-1, 1)).reshape(-1)

        print(np.mean(pred_cont), np.std(pred_cont))


if __name__ == "__main__":
    params = get_params()
    print(params)
    main(params)
