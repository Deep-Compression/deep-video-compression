import os
import ffmpeg

from config import *

for model in MODELS:
    total_file_size = 0
    num_files = 0

    for root, _, files in os.walk('../results/output/mp4/' + model + '/depth_0'):
        for file in files:
            if file.endswith('.mp4'):
                probe = ffmpeg.probe(root + '/' + file)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

                print(root, file)
                print(int(video_stream['bit_rate']) / 25 / (256 * 448))

                input()

                #total_file_size += os.path.getsize(root + '/' + file)
                #num_files += 1

    mean_file_size = total_file_size / num_files
    mean_bpp = mean_file_size / (256 * 448) / 8

    print('model', model, '\ntotal file size', total_file_size, '\nnum files', num_files, '\nmean file size',
          mean_file_size, '\nmean bpp', mean_bpp)

    print()

"""

    HiFiC
    
       | 0                     | 1                     | 2
    ---+-----------------------+-----------------------+----------------------
    hi | 0.006066620178241761  | 0.0030333100891208804 | 0.0015166550445604402
    mi | 0.004331266714363294  | 0.002165633357181647  | 0.0010828166785908234
    lo | 0.0024148429516216417 | 0.0012074214758108208 | 0.0006037107379054104
    
    0.30228585379464284
    
    JPEG
    
       | 0                    | 1                     | 2
    ---+----------------------+-----------------------+----------------------
    hi | 0.006192135435726621 | 0.0030960677178633106 | 0.0015480338589316553
    mi | 0.004539023436328255 | 0.0022695117181641277 | 0.0011347558590820638
    lo | 0.003173811925949594 | 0.001586905962974797  | 0.0007934529814873985
    
    JPEG similar bit rates without interpolation
    
       | 1                     | 2
    ---+-----------------------+-----------------------
    hi | 0.0035359527226814986 | 0.003090625996134137
    mi | 0.003116204102694762  | 0.003090562578045831
    lo | 0.003090925232462577  | 0.003090562578045831
    
    MP4
    
       | 0                     | 1                     | 2
    ---+-----------------------+-----------------------+----------------------
    hi |  |  | 
    mi |  |  | 
    lo |  |  | 

"""



"""
    BYTES PER IMAGE -> BITRATE (BITS / SECOND, 25 FPS)

    interpolation depth: 1
    
        hific-hi: 2783.0741400087640 -> 556614.82800175280
        hific-mi: 1986.9772677475898 -> 397395.45354951796
        hific-lo: 1107.8140337423313 -> 221562.80674846625

    interpolation depth: 2
    
        hific-hi: 1391.537070004382 -> 278307.41400087640
        hific-mi: 993.4886338737949 -> 198697.72677475898
        hific-lo: 553.9070168711656 -> 110781.40337423312
"""



