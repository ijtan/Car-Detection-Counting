import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from object_detection import *

video_path='videos'


def get_video_paths():
    """
    Returns a list of all the videos in the directory
    """
    # get nested videos in the directory mp4 or avi
    videos = []
    for dir in os.listdir(video_path):
        if os.path.isdir(os.path.join(video_path, dir)):
            for file in glob.glob(os.path.join(video_path, dir, '*.mp4')):
                videos.append(file)
    return videos

def label_videos(videos, show_video=False):
    """
    Labels the videos in the directory
    """
    for video_path in videos:
        video = cv2.VideoCapture(video_path)
        video_name = video_path.split('/')[-1]
        print('Labelling video: {}'.format(video_name))
        
        labelled_frames = []

        while True:
            _, frame = video.read()
            if frame is None:
                break

            labelled_frames.append(image_detect_loaded(frame))
            if show_video:
                cv2.imshow('Video', frame)
                cv2.waitKey(0)

        video.release()
        cv2.destroyAllWindows()
        cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
        


if __name__ == '__main__':
    videos = get_video_paths()

    
    print('Found {} videos'.format(len(videos)))

    example_path = videos[-3]
    label_videos([example_path], True)
    # labels = get_video_labels(videos)
    # print(labels)