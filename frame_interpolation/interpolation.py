"""
    Created on 18 Feb 2021 10:17am

    @author Felix Beutter
"""
from multiprocessing import Manager, Process

from helper.print_progress_bar import print_progress_bar


def interpolate_intermediate_frames(first_frame, second_frame, method, depth):
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
            intermediate_frames.extend(method(frames[i], frames[i + 1]))

        merged_frames = [None] * (len(frames) + len(intermediate_frames))
        merged_frames[::2] = frames
        merged_frames[1::2] = intermediate_frames

        frames = merged_frames

    return frames[1:-1]


def interpolate_frames(key_frames, num_end_frames, method, depth, process_dict=None, print_log_messages=True,
                       print_progress=True):
    """
        Interpolates intermediate frames between a series of key frames.

        :param key_frames:
        :param num_end_frames: Number of end frames (key frames at the end which will have no interpolated intermediate
                               frames
        :param method: Interpolation method (e.g. 'linear_interpolation' or 'sepconv_slomo_interpolation')
        :param depth: Recursion steps (determines the number of interpolated intermediate frames)
        :param process_dict: Process dictionary to store results if method is executed in isolated process
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: All frames including key and interpolated intermediate frames
    """
    if print_log_messages:
        print('Interpolating intermediate frames...')

    if print_progress:
        print_progress_bar(0, len(key_frames) - num_end_frames)

    frames = []

    for n in range(len(key_frames) - num_end_frames):
        frames.append(key_frames[n])

        interpolated_frames = interpolate_intermediate_frames(key_frames[n], key_frames[n + 1], method, depth)
        frames.extend(interpolated_frames)

        if print_progress:
            print_progress_bar(n + 1, len(key_frames) - num_end_frames)

    frames.extend(key_frames[-num_end_frames:])

    if process_dict is not None:
        process_dict['return'] = frames

    return frames


def interpolate_frames_process(key_frames, num_end_frames, method, depth, print_log_messages=True, print_progress=True):
    """
        Interpolates intermediate frames between a series of key frames (executed in an isolated process).

        :param key_frames:
        :param num_end_frames: Number of end frames (key frames at the end which will have no interpolated intermediate
                               frames
        :param method: Interpolation method (e.g. 'linear_interpolation' or 'sepconv_slomo_interpolation')
        :param depth: Recursion steps (determines the number of interpolated intermediate frames)
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: All frames including key and interpolated intermediate frames
    """
    process_dict = Manager().dict()
    process = Process(target=interpolate_frames,
                      args=[key_frames, num_end_frames, method, depth, process_dict, print_log_messages,
                            print_progress])

    process.start()
    process.join()

    return process_dict['return']
