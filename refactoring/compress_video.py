"""
    Created on 18 Feb 2021 09:42am

    @author Felix Beutter
"""
import cv2
import pickle

from .image_compression import compress_frames_hific


def compress_video(input_file, output_file='compressed_video.dvc', model='hific-lo', interpolation=None,
                   interpolation_depth=1):
    """
        Compresses a video file using the hific generative image compression.

        :param input_file: Video to compress
        :param output_file: File name of compressed video
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :param interpolation: Method to interpolate frames for further data reduction (None for no interpolation)
        :param interpolation_depth: Determines the number of frames to interpolate between to compressed ones
    """
    video_capture = cv2.VideoCapture(input_file)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames < 1:
        raise RuntimeError('ERROR: Video has no frames, compression will not be performed.')

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    num_end_frames = (num_frames - 1) % (interpolation_depth ** 2) + 1

    dictionary = {'fps': fps, 'model': model, 'num_frames': num_frames, 'interpolation': interpolation,
                  'interpolation_depth': interpolation_depth, 'num_end_frames': num_end_frames}

    frames_to_compress = []
    _, cv_image = video_capture.read()

    for n in range(num_frames):
        if interpolation is None or n % (interpolation_depth ** 2) == 0 or n >= num_frames - num_end_frames:
            frames_to_compress.append(cv_image)

        _, cv_image = video_capture.read()

    packed_tensors = compress_frames_hific(frames_to_compress, model)
    dictionary['packed_tensors'] = packed_tensors

    pickle.dump(dictionary, open(output_file, 'wb'))
