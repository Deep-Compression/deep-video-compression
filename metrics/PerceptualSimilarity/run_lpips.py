import numpy as np
import torch


def calculate_lpips(img0, img1, model):
    img0 = np.expand_dims(img0, 0)
    img1 = np.expand_dims(img1, 0)

    img0 = np.array(img0).transpose([0, 3, 1, 2])
    img1 = np.array(img1).transpose([0, 3, 1, 2])

    # convert to torch tensor
    img0 = torch.from_numpy(img0)
    img1 = torch.from_numpy(img1)

    # normalize images to [-1, 1]
    img0 = img0 / 127.5 - 1
    img1 = img1 / 127.5 - 1

    return model(img0, img1)
