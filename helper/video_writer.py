"""
    Created on 18 Feb 2021 11:41am

    @author Felix Beutter
"""
import cv2
import numpy as np
from pathlib import Path


def write_frames_to_video(file_name, frames, fps, fourcc='mp4v', print_log_messages=True):
    """
        Generates a video out of frames and writes it to file (as .mp4).

        :param file_name: File name of the video
        :param frames: Frames
        :param fps: Frames per second
        :param fourcc: Video codec
        :param print_log_messages: If True, log message are printed to console
    """
    if print_log_messages:
        print('Writing video to file (\'{}\')... '.format(file_name), end='', flush=True)

    Path('/'.join(file_name.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

    frame_shape = np.shape(frames[0])
    width, height = frame_shape[1], frame_shape[0]

    video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()

    if print_log_messages:
        print('done')
