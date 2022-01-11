import os
from pathlib import Path
import cv2


def get_video_paths(video_path):
    """
    Recursively Returns a list of all the videos in the directory
    """
    return list(Path.rglob(Path(video_path), '*.mp4'))

def count_frames(paths):
    """
    Counts the number of frames in a video
    """
    count = 0
    for path in paths:
        video = cv2.VideoCapture(path)
        
        while True:
            _, frame = video.read()
            if frame is None:
                break
            count += 1
    return count