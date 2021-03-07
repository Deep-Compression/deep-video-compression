"""
    Created on 20 Feb 2021 06:49pm

    @author Felix Beutter
"""
import cv2

from helper.video_writer import write_frames_to_video


def split_video_in_clips(input_file, output_dir, clip_length, fps):
    """
        Splits video of into several clips of specific length.

        :param input_file: Input video file
        :param output_dir: Output directory
        :param clip_length: Length of each generated clip (in seconds)
        :param fps: Frames per second
    """
    video_capture = cv2.VideoCapture(input_file)
    success, cv_image = video_capture.read()

    frames = []
    n = 0

    while success:
        frames.append(cv_image)

        if len(frames) == int(clip_length * fps):
            write_frames_to_video(output_dir + '/clip_{}.avi'.format(n), frames, fps,  'RGBA')

            frames = []
            n += 1

        success, cv_image = video_capture.read()

    if len(frames) > 0:
        write_frames_to_video(output_dir + '/clip_{}.avi'.format(n), frames, fps, 'RGBA')
