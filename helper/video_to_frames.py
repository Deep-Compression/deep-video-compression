"""
    Created on 22 Feb 2021 03:59pm

    @author Felix Beutter
"""
import cv2
import numpy as np


def video_to_frames(file):
    """
        Converts a video file into a list of frames.

        :param file: Video file
        :return: Frames
    """
    video_capture = cv2.VideoCapture(file)
    success, cv_image = video_capture.read()

    frames = []

    while success:
        frames.append(cv_image)
        success, cv_image = video_capture.read()

    return frames
