import lpips
import torch

def calculate_lpips(img0, img1):
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    #loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    
    # convert to torch tensor
    img0 = torch.from_numpy(img0)
    img1 = torch.from_numpy(img1)

    # normalize images to [-1, 1] !!!
    img0 = img0 / 127.5 - 1
    img1 = img1 / 127.5 - 1
    
    return loss_fn_alex(img0, img1)