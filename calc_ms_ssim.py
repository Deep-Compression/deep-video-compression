import numpy as np
from PIL.Image import Image

from helper.multi_scale_ssim import multi_scale_ssim


def calculate_ms_ssim(original_image, compared_image):
    """
        Calculates MS-SSIM between two images.

        :param original_image: Original image file
        :param compared_image: Compared image file
    """
    original_image = np.asarray(Image.open(original_image))
    compared_image = np.asarray(Image.open(compared_image))

    original_image = np.expand_dims(original_image, axis=0)
    compared_image = np.expand_dims(compared_image, axis=0)

    return multi_scale_ssim(original_image, compared_image)
