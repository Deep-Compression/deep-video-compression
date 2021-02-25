"""
    Created on 18 Feb 2021 09:51am

    @author Felix Beutter
"""
import pickle

from frame_interpolation.interpolation import interpolate_frames_process
from frame_interpolation.linear import linear_interpolation
from frame_interpolation.sepconv_slomo import sepconv_slomo_interpolation
from helper.video_writer import write_frames_to_video
from image_compression.hific_decompression import decompress_process


def decompress_dvc(input_file_or_dict, output_file='decompressed_video.mp4', interpolation='linear',
                   print_log_messages=True, print_progress=True):
    """
        Decompresses a compressed video file using the hific generative image compression (and frame interpolation if
        applicable).

        :param input_file_or_dict: Compressed video to decompress (either file name to load dict from or dict itself)
        :param output_file: File name of decompressed video (None, if the video should not be written to file)
        :param interpolation: Method to interpolate frames for further data reduction
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: Frames of decompressed video
    """
    if type(input_file_or_dict) is not dict:
        dictionary = pickle.load(open(input_file_or_dict, 'rb'))

    else:
        dictionary = input_file_or_dict

    decompressed_frames = decompress_process(dictionary['packed_tensors'], dictionary['model'], print_log_messages,
                                             print_progress)

    if interpolation == 'linear':
        method = linear_interpolation

    elif interpolation == 'sepconv_slomo':
        method = sepconv_slomo_interpolation

    else:
        raise RuntimeError('Invalid interpolation method (\'{}\').'.format(interpolation))

    frames = interpolate_frames_process(decompressed_frames, dictionary['num_end_frames'], method,
                                        dictionary['interpolation_depth'], print_log_messages, print_progress)

    if output_file is not None:
        write_frames_to_video(output_file, frames, dictionary['fps'], print_log_messages=print_log_messages)

    return frames
