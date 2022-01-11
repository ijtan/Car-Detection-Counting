import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from object_detection import *
from tqdm import tqdm

video_path='inputs/media/Clips/'
output_txt_path = 'outputs/yolo_coco_classification/'


def get_video_paths():
    """
    Recursively Returns a list of all the videos in the directory
    """
    return list(Path.rglob(Path(video_path), '*.mp4'))
    


def label_videos(videos, show_video=False, scale=1):
    """
    Labels the videos in the directory
    """
    for video_path in videos:
        video = cv2.VideoCapture(video_path)
        
        # downsize video to scale
        if scale != 1:
            video.set(cv2.CAP_PROP_FRAME_WIDTH, int(video.get(cv2.CAP_PROP_FRAME_WIDTH) / scale))
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) / scale))

        # if remove_same_frames:
        #     video = removeDuplicateFrames(video)

        video_name = video_path.split('/')[-1]
        coord_path  = os.path.join(output_txt_path, video_name)

        print('Labelling video: {}'.format(video_name))
        
        labelled_frames = []
        while True:
            _, frame = video.read()
            if frame is None:
                break

            print('Labelling frame: {}'.format(video.get(cv2.CAP_PROP_POS_FRAMES)))
            
            (image_detect_loaded(frame, video.get(cv2.CAP_PROP_POS_FRAMES), coord_path))

            if show_video:
                cv2.imshow('Video', labelled_frames[-1])
                # cv2.waitKey(0)
                # wait for a few ms
                cv2.waitKey(1)
            prev_frame = frame

        video.release()
        cv2.destroyAllWindows()
        #cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
        
        


if __name__ == '__main__':
    videos = get_video_paths()
    print('Found {} total videos'.format(len(videos)))

    test_videos = [str(v) for v in videos if 'test' in str(v).lower()]
    print('Found {} test videos'.format(len(test_videos)))

    label_videos(test_videos, show_video=False, scale=1)