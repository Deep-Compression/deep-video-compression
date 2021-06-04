import os
import cv2

from pathlib import Path

from config import *

def run():
    for bitrate in [556615, 397395, 221563, 278307, 198698, 110781]:
        for root, _, files in os.walk('../dataset_tiny'):
            if 'im1.png' in files:

                # MP4
                out_root = 'output/mp4/' + str(bitrate) + root[10:]
                Path(out_root).mkdir(parents=True, exist_ok=True)

                print('Encoding ' + out_root + '.mp4')

                os.system('ffmpeg -loglevel quiet -y -i {root_dict}/im%1d.png -c:v libx264 \
                    -b:v {bitr} -pass 1 -an -f null /dev/null && \
                    ffmpeg -loglevel quiet -i {root_dict}/im%1d.png -c:v libx264 \
                    -b:v 2600k -pass 2 -c:a aac -b:a 128k -pix_fmt yuv420p \
                    -y {out_file}.mp4 -loglevel quiet'.format(root_dict=root, out_file=out_root, bitr=str(bitrate)))

                video_to_frames(out_root, '.mp4')

                # AV1
                out_root = 'output/av1/' + str(bitrate) + root[10:]
                Path(out_root).mkdir(parents=True, exist_ok=True)

                print('Encoding ' + out_root + '.mkv')

                os.system('ffmpeg -loglevel quiet -i {root_dict}/im%1d.png -c:v libaom-av1 \
                    -b:v {bitr} -pass 1 -an -f null /dev/null && \
                    ffmpeg -loglevel quiet -i {root_dict}/im%1d.png -c:v libaom-av1 -b:v 2M \
                    -pass 2 -c:a libopus -pix_fmt yuv420p \
                    -y {out_file}.mkv'.format(root_dict=root, out_file=out_root, bitr=str(bitrate)))
                
                video_to_frames(out_root, '.mkv')


def video_to_frames(out_root, suffix):
    cam = cv2.VideoCapture(out_root + suffix)
    frame_index = 1

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        file_name = out_root + '/im' + str(frame_index) + '.png'
        cv2.imwrite(file_name, frame)

        frame_index += 1

if __name__ == '__main__':
    run()