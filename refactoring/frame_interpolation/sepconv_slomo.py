"""
    Created on 18 Feb 2021 10:13am

    @author Felix Beutter
"""
import torch
import numpy as np

from ..sepconv_slomo import estimate


def sepconv_slomo_interpolation(first_frame, second_frame):
    """
        Generates one intermediate frame between two frames using sepconv slomo.

        :param first_frame: First frame
        :param second_frame: Second frame
        :return: Interpolated intermediate frames
    """
    first_frame_data = np.array(first_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    first_tensor = torch.FloatTensor(np.ascontiguousarray(first_frame_data))

    second_frame_data = np.array(second_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    second_tensor = torch.FloatTensor(np.ascontiguousarray(second_frame_data))

    output = estimate(first_tensor, second_tensor)
    return (output.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)
