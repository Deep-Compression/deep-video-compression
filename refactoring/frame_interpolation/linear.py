"""
    Created on 18 Feb 2021 10:08am

    @author Felix Beutter
"""
import numpy as np


def linear_interpolation(first_frame, last_frame, num_intermediate_frames=1):
    """
        Generates intermediate frames between a first and last frame using linear interpolation.

        :param first_frame: First frame
        :param last_frame: Last frame
        :param num_intermediate_frames: Number of intermediate frames to be generated
        :return: Numpy array of generated intermediate frames
    """
    return np.linspace(first_frame, last_frame, num_intermediate_frames + 2, dtype=np.uint8)[1:-1]
