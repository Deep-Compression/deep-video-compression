import os
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from .model.RIFE_HDv2 import Model
import warnings

def inference(image0, image1, exp):
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = Model()
    model.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log'), -1)
    model.eval()
    model.device()

    #img0 = np.asmatrix(image0)
    #img1 = np.asmatrix(image1)
    img0 = (torch.tensor(image0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(image1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    out = []
    for i in range(len(img_list)):
        out.append((img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    
    return out[1:-1]
