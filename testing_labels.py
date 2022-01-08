Day_1 = "Gozo Ferry/footage2"
Night_1 = "Gozo Ferry/nightfootage2"
Rain_1 = "Gozo Ferry/rainTest"
Day_2 = "Gozo Ferry/footage3"
Night_2 = "Gozo Ferry/20190127_200452"

from main import *
from object_detection import *
import os
import glob

vid_path = 'videos/'

rain_1_path = vid_path + Rain_1 + '.mp4'

def show_video_labels(vid_path, box_dirs):
    box_files = glob.glob(vid_path + '*.txt')
    video = cv2.VideoCapture(vid_path)
    
    while True:
        _, frame = video.read()
        if frame is None:
            break

        boxes = []

        # get box file by frame index
        box_file = box_files[video.get(cv2.CAP_PROP_POS_FRAMES)]
        with open(box_file, 'r') as f:
            for line in f:
                coords = line.split(' ')
                index = coords[0]
                coords = [float(x) for x in coords[1:]]

                boxes.append(coords)

        label_image(frame, boxes)
        cv2.imshow('video', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    video.release()
    cv2.destroyAllWindows()
        


