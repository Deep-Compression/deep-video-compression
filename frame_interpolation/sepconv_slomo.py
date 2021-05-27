"""
    Created on 18 Feb 2021 10:13am

    @author Felix Beutter
"""
import torch
import numpy as np

from sepconv_slomo.run import estimate


def sepconv_slomo_interpolation(first_frame, second_frame, depth):
    """
        Generates one intermediate frame between two frames using sepconv slomo.

        :param first_frame: First frame
        :param second_frame: Second frame
        :return: Interpolated intermediate frame
    """

    if depth not in [1, 2]:
        raise ValueError("Interpolation depth must be 1 or 2")

    first_frame_data = np.array(first_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    first_tensor = torch.FloatTensor(np.ascontiguousarray(first_frame_data))

    second_frame_data = np.array(second_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    second_tensor = torch.FloatTensor(np.ascontiguousarray(second_frame_data))

    output = estimate(first_tensor, second_tensor)

    output = [(output.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)]
    
    if depth == 2:
        middle_frame_data = output[0][:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        middle_tensor = torch.FloatTensor(np.ascontiguousarray(middle_frame_data))
        
        output_left = estimate(first_tensor, middle_tensor)
        output_right = estimate(middle_tensor, second_tensor)

        output_left = (output_left.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)
        output_right = (output_right.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)

        output.insert(0, output_left)
        output.append(output_right)

    return output
