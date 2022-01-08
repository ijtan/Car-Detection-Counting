import cv2
import numpy as np 
import argparse
import time
import os

weights_path = 'models/yolov3.weights'
config_path = 'models/yolov3.cfg'
names_path = 'models/coco.names'

def load_yolo():
	net = cv2.dnn.readNet(weights_path, config_path)
	classes = []
	with open(names_path, "r") as f:
		classes = [line.strip() for line in f.readlines()] 
	
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs


def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			# print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	# cv2.imshow("Image", img)
	return img


def draw_txt_labels(boxes, colors, img): 
	# set all confs as 100
	confs = [100]*len(boxes)
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			image_x_res = img.shape[1]
			image_y_res = img.shape[0]

			x = round(x*image_x_res)
			y = round(y*image_y_res)
			w = round(w*image_x_res)
			h = round(h*image_y_res)
			
			label = str('car')
			print(label, x, y, w, h)
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	# cv2.imshow("Image", img)
	return img


# def image_detect(img_path): 
# 	model, classes, colors, output_layers = load_yolo()
# 	image, height, width, channels = load_image(img_path)
# 	blob, outputs = detect_objects(image, model, output_layers)
# 	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
# 	labelled = draw_labels(boxes, confs, colors, class_ids, classes, image)
# 	# while True:
# 	# 	key = cv2.waitKey(1)
# 	# 	if key == 27:
# 	# 		break
# 	return labelled

def append_box_to_file(path,frame_no, boxes):
	# box_ax_csv = [box[0], box[1], box[2], box[3]]
	str_to_append = ""
	for ix, coords in enumerate(boxes):
		str_to_append += str(0)+' '+str(coords[0])+' '+str(coords[1])+' '+str(coords[2])+' '+str(coords[3])+'\n'

	new_path = os.path.join(path, str(frame_no)+'.txt')

	# create if not exists
	if not os.path.exists(new_path):
		# if subdirs do not exist, create aswell
		if not os.path.exists(os.path.dirname(new_path)):
			os.makedirs(os.path.dirname(new_path))

		with open(new_path, 'w') as f:
			f.write(str_to_append)
	else:
		with open(new_path, 'a') as f:
			f.write(str_to_append)



model, classes, colors, output_layers = load_yolo()
def image_detect_again(img_path): 
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	labelled = draw_labels(boxes, confs, colors, class_ids, classes, image)
	return labelled


def image_detect_loaded(image, frame_no, path):
	height, width, channels = image.shape
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

	normalized_boxes = []
	for box in boxes:
		normalized_boxes.append([box[0]/width, box[1]/height, box[2]/width, box[3]/height])
	append_box_to_file(path, frame_no, normalized_boxes)

	labelled = draw_labels(boxes, confs, colors, class_ids, classes, image)
	return labelled

def label_image(image, boxes):
	
	labelled = draw_txt_labels(boxes,colors, image)
	return labelled









# def webcam_detect():
# 	model, classes, colors, output_layers = load_yolo()
# 	cap = start_webcam()
# 	while True:
# 		_, frame = cap.read()
# 		height, width, channels = frame.shape
# 		blob, outputs = detect_objects(frame, model, output_layers)
# 		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
# 		draw_labels(boxes, confs, colors, class_ids, classes, frame)
# 		key = cv2.waitKey(1)
# 		if key == 27:
# 			break
# 	cap.release()


# def start_video(video_path):
# 	model, classes, colors, output_layers = load_yolo()
# 	cap = cv2.VideoCapture(video_path)
# 	while True:
# 		_, frame = cap.read()
# 		height, width, channels = frame.shape
# 		blob, outputs = detect_objects(frame, model, output_layers)
# 		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
# 		draw_labels(boxes, confs, colors, class_ids, classes, frame)
# 		key = cv2.waitKey(1)
# 		if key == 27:
# 			break
# 	cap.release()