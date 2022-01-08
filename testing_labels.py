Day_1 = "Gozo Ferry/footage2"
Night_1 = "Gozo Ferry/nightfootage2"
Rain_1 = "Gozo Ferry/rainTest"
Day_2 = "Gozo Ferry/footage3"
Night_2 = "Gozo Ferry/20190127_200452"

msida = "Msida/20200323_155250.mp4"

from main import *
from object_detection import *
import os
import glob



def show_video_labels(vid_path, box_dir):
    box_files = glob.glob(box_dir + '*.txt')
    video = cv2.VideoCapture(vid_path)
    
    while True:
        _, frame = video.read()
        if frame is None:
            break

        boxes = []

        # get box file by frame index
        box_file = box_files[int(video.get(cv2.CAP_PROP_POS_FRAMES))-1]
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
        
if __name__ == '__main__':
    vid_path = 'videos/'
    # rain_1_path = vid_path + Rain_1 + '.mp4'
    # rain_1_boxes = 'rain_1_yolo/'

    msida_vid = vid_path + msida
    msida_boxes = "output_coords/videos/Msida/20200323_155250.mp4/"

    show_video_labels(msida_vid, msida_boxes)
