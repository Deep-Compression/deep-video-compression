import sys
import cv2
import numpy as np

from helper.multi_scale_ssim import multi_scale_ssim
from helper.print_progress_bar import print_progress_bar


def calc_mean_ms_ssim(original_video, compared_video):
    """
        Calculates the mean MS-SSIM between two videos.

        :param original_video: Original video file
        :param compared_video: Compared video file
        :return: Mean MS-SSIM over all video frames
    """
    print('Calculating mean MS-SSIM of \'{}\' and \'{}\'...'.format(original_video, compared_video))

    original_video_capture = cv2.VideoCapture(original_video)
    compared_video_capture = cv2.VideoCapture(compared_video)

    num_frames = int(original_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames != int(compared_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise RuntimeError('ERROR! Videos does not have the same number of frames. Only videos with the same number of '
                           'frames can be compared.')

    total_ms_ssim = 0.
    print_progress_bar(0, num_frames)

    for i in range(num_frames):
        _, original_image = original_video_capture.read()
        original_image = np.expand_dims(original_image, axis=0)

        _, compared_image = compared_video_capture.read()
        compared_image = np.expand_dims(compared_image, axis=0)

        total_ms_ssim += multi_scale_ssim(original_image, compared_image)
        print_progress_bar(i + 1, num_frames)

    ms_ssim = total_ms_ssim / num_frames
    print('Mean MS-SSIM: {}'.format(ms_ssim))

    return ms_ssim


def main():
    """
        Main method. Parses arguments and calls mean_ms_ssim method.

        Usage: python mean_ms_ssim.py [original video file] [compared video file]
    """
    if len(sys.argv) != 3:
        raise RuntimeError('ERROR! Invalid number of arguments.')

    calc_mean_ms_ssim(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
