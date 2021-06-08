import os
import cv2

from pathlib import Path

from helper.print_progress_bar import print_progress_bar
from config import *

'''
OpenCV: Couldn't read video stream from file "output/mp4/hific-mi/depth_2/00036/0974.mp4"

OpenCV: Couldn't read video stream from file "output/mp4/hific-mi/depth_2/00091/0021.mp4"

OpenCV: Couldn't read video stream from file "output/mp4/hific-lo/depth_2/00036/0974.mp4"

OpenCV: Couldn't read video stream from file "output/mp4/hific-lo/depth_2/00091/0472.mp4"

OpenCV: Couldn't read video stream from file "output/mp4/hific-lo/depth_2/00091/0021.mp4"
'''

def run():
    for model in MODELS:
        for depth in [0]:
            if depth == 1:
                dc_indices = [1, 3, 5]
            elif depth == 2:
                dc_indices = [1]
            elif depth == 0:
                dc_indices = [1, 2, 3, 4, 5, 6, 7]
            counter = 0
            for root, _, files in os.walk(DATASET_DIR):
                if 'im1.png' in files:
                    if counter < 1200:
                        counter += 1
                        print_progress_bar(counter, 1200, prefix='Progress model: {} depth: {}'.format(model, depth))

                        bitrate = get_bitrate_for_dvc(model, root, dc_indices, depth)

                        for codec in ['mp4', 'av1']:
                            out_root = 'output/' + codec + '/' + model + '/depth_' + str(depth) + root[10:]
                            Path(out_root).mkdir(parents=True, exist_ok=True)

                            frames_to_video(root, out_root, codec, bitrate)
                            if depth == 1:
                                video_to_frames(out_root, codec, 6)
                            elif depth == 2:
                                video_to_frames(out_root, codec, 4)
                            elif depth == 0:
                                video_to_frames(out_root, codec, 7)
                    else:
                        break


def get_bitrate_for_dvc(model, root, dc_indices, depth):
    dc_files_size = 0
    for dc_root, _, _ in os.walk(COMPRESSED_DIR + '/' + model + root[10:]):
        for dc_index in dc_indices:
            dc_files_size += os.path.getsize(dc_root + '/im' + str(dc_index) + '.dc')

    if depth == 1:
        bytes_per_image = dc_files_size / 6
    elif depth == 2:
        bytes_per_image = dc_files_size / 4
    elif depth == 0:
        bytes_per_image = dc_files_size / 7
        
    return int(bytes_per_image * 25 * 8)


def frames_to_video(root, out_root, codec, bitrate):
    if codec not in ['mp4', 'av1']:
        raise ValueError('Codec must be mp4 or av1')

    if codec == 'mp4':
        os.system('ffmpeg -loglevel quiet -y -i {root_dict}/im%1d.png -c:v libx264 \
            -b:v {bitr} -pass 1 -an -f null /dev/null && \
            ffmpeg -loglevel quiet -y -i {root_dict}/im%1d.png -c:v libx264 \
            -b:v {bitr} -pass 2 -pix_fmt yuv420p \
            {out_file}.mp4 -loglevel quiet'.format(root_dict=root, out_file=out_root, bitr=str(bitrate)))

    if codec == 'av1':
        os.system('ffmpeg -loglevel quiet -y -i {root_dict}/im%1d.png -c:v libaom-av1 \
            -b:v {bitr} -pass 1 -an -f null /dev/null && \
            ffmpeg -loglevel quiet -y -i {root_dict}/im%1d.png -c:v libaom-av1 -b:v 2M \
            -b:v {bitr} -pass 2 -pix_fmt yuv420p \
            {out_file}.mkv'.format(root_dict=root, out_file=out_root, bitr=str(bitrate)))


def video_to_frames(out_root, codec, num_frames):
    if codec == 'av1':
        suffix = '.mkv'
    elif codec == 'mp4':
        suffix = '.mp4'

    cam = cv2.VideoCapture(out_root + suffix)

    for frame_index in range(1, num_frames + 1):
        ret, frame = cam.read()

        if not ret:
            break

        file_name = out_root + '/im' + str(frame_index) + '.png'
        cv2.imwrite(file_name, frame)

if __name__ == '__main__':
    run()
