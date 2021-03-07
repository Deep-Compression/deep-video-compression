# this is an implementation of the peak signal-to-noise ratio metric found on the following site:
# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

import math
import numpy as np

def calculate_psnr(img0, img1):
    # img1 and img2 have range [0, 255]
    img0 = img0.astype(np.float64)
    img1 = img1.astype(np.float64)
    mse = np.mean((img0 - img1)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
