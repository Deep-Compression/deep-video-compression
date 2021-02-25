"""
    Created on 18 Feb 2021 10:17am

    @author Felix Beutter
"""


def interpolate_frames(first_frame, second_frame, method, depth):
    """
        Uses an interpolation method to recursively interpolate intermediate images between two given frames.

        :param first_frame: First frame
        :param second_frame: Second frame
        :param method: Interpolation method (e.g. 'linear_interpolation' or 'sepconv_slomo_interpolation')
        :param depth: Recursion steps (determines the number of interpolated intermediate frames)
        :return: List of interpolated intermediate frames
    """
    frames = [first_frame, second_frame]

    for _ in range(depth):
        intermediate_frames = []

        for i in range(len(frames) - 1):
            intermediate_frames.append(method(frames[i], frames[i + 1]))

        merged_frames = [None] * (len(frames) + len(intermediate_frames))
        merged_frames[::2] = frames
        merged_frames[1::2] = intermediate_frames

        frames = merged_frames

    return frames[1:-1]
