import os
import ffmpeg

from config import *

for model in MODELS:
    total_bpp = 0

    total_file_size = 0
    num_files = 0

    for root, _, files in os.walk('../results/output/av1/' + model + '/depth_0'):
        for file in files:
            if file.endswith('.mkv'):
                print(root + '/' + file)
                probe = ffmpeg.probe(root + '/' + file)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

                total_bpp += int(video_stream['bit_rate']) / 25 / (256 * 448)

                #total_file_size += os.path.getsize(root + '/' + file)
                num_files += 1

    mean_bpp = total_bpp / num_files

    #mean_file_size = total_file_size / num_files
    #mean_bpp = mean_file_size / (256 * 448) * 8

    print('model', model, '\nmean bpp', mean_bpp)

    print()

"""
    
    HiFiC
    
       | 0                   | 1                   | 2
    ---+---------------------+---------------------+----------------------
    hi | 0.3882636914074727  | 0.19413184570373634 | 0.09706592285186817
    mi | 0.2772010697192508  | 0.1386005348596254  | 0.0693002674298127
    lo | 0.15454994890378507 | 0.07727497445189253 | 0.03863748722594627
    
    JPEG
    
       | 0                    | 1                     | 2
    ---+----------------------+-----------------------+----------------------
    hi | 0.39629666788650375 | 0.19814833394325188 | 0.09907416697162594
    mi | 0.29049749992500834 | 0.14524874996250417 | 0.07262437498125209
    lo | 0.203123963260774   | 0.101561981630387   | 0.0507809908151935
    
    JPEG similar bit rates without interpolation
    
       | 1                   | 2
    ---+---------------------+-----------------------
    hi | 0.2263009742516159  | 0.19780006375258477
    mi | 0.19943706257246477 | 0.1977960049949332
    lo | 0.19781921487760493 | 0.1977960049949332
    
    MP4
    
       | 0                   | 1                     | 2
    ---+---------------------+-----------------------+----------------------
    hi | 0.3777033142089845  | 0.19196396193731396 | 0.09667959129696822
    mi | 0.272490527925037   | 0.13757831653413327 | 0.06941503719927579
    lo | 0.15286509806315096 | 0.07752288498651423 | 0.038540334805111855

    AV1
    
       | 0                   | 1                     | 2
    ---+---------------------+-----------------------+----------------------
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



