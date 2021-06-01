import os
import cv2
from pathlib import Path
from config import *


# install ffmpeg:
# sudo apt install ffmpeg
def encode_to_video():
    for root, _, files in os.walk(DATASET_DIR):
        if 'im1.png' in files:

            #create paths
            out_root = COMPRESSED_DIR + '/mp4' + root[10:]
            Path(out_root).mkdir(parents=True, exist_ok=True)

            # encode in H.264 / MP4 -> https://trac.ffmpeg.org/wiki/Encode/H.264
            os.system('ffmpeg -y -i {root_dict}/im%1d.png -c:v libx264 -b:v 2600k -pass 1 -an -f null /dev/null && \
                ffmpeg -i {root_dict}/im%1d.png -c:v libx264 -b:v 2600k -pass 2 -c:a aac -b:a 128k -pix_fmt yuv420p -y {out_file}.mp4'.format(root_dict = root, out_file = out_root))

            # convert back to png
            convert_back_to_png(out_root, '.mp4')

            #create paths
            out_root = COMPRESSED_DIR + '/av1' + root[10:]
            Path(out_root).mkdir(parents=True, exist_ok=True)

            # encode in AV1 -> https://trac.ffmpeg.org/wiki/Encode/AV1
            os.system('ffmpeg -i {root_dict}/im%1d.png -c:v libaom-av1 -b:v 2M -pass 1 -an -f null /dev/null && \
                ffmpeg -i {root_dict}/im%1d.png -c:v libaom-av1 -b:v 2M -pass 2 -c:a libopus -pix_fmt yuv420p -y {out_file}.mkv'.format(root_dict = root, out_file = out_root))
            
            # convert back to png
            convert_back_to_png(out_root, '.mkv')


def convert_back_to_png(out_root, container):
    cam = cv2.VideoCapture(out_root + container)
    currentframe = 1
    while(True):
        ret,frame = cam.read()
        if ret:
            name = out_root + '/im' + str(currentframe) + '.png'
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

if __name__ == '__main__':
    encode_to_video()
