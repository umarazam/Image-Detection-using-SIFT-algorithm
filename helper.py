# helper.py

import pickle
import cv2
from glob import glob

def disImage(images):
	import numpy as np
	stack = np.hstack((images))
	cv2.imshow('Images',stack)
	cv2.waitKey()

def load_images(path):
	try:
		filenames = glob(path)
		filenames.sort()
		images= [cv2.imread(img,0) for img in filenames] # --> 0 will convert image to gray scale at the start
		return images
	except Exception:
		print(f"Couldn't Load the images with path {path}")

def showImg(img):
	cv2.imshow('Image', img)
	cv2.waitKey(0)

def showImg_kp(im, kp):
	out_img = cv2.drawKeypoints(im, kp, None)
	cv2.imshow('Image',out_img)
	cv2.waitKey(0)

def keypoint_load(path):
	kp = []
	pickle_in = open(path, 'rb')
	data = pickle.load(pickle_in, encoding='bytes')
	for d in data:
		temp_kp = []
		for p in d:
			temp = (cv2.KeyPoint(x=p[0][0],y=p[0][1],_size=p[1],_angle=p[2],_response=p[3],_octave=p[4],_class_id = p[5]))
			temp_kp.append(temp)
		kp.append(temp_kp)
	return kp

def keypoint_save(keypoints,path):
	try:
		arr =[]
		print(len(keypoints))
		for point in keypoints:
			temp_kp = []
			for kp in point:
				temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
				temp_kp.append(temp)
			arr.append(temp_kp)
		print(pickle_save(arr, path))
	except Exception:
		print("Error in saving keypoints")

def pickle_load(path):
	ds = []
	p = open(path,'rb')
	obj = pickle.load(p)
	for ob in obj:
		ds.append(ob)
	return ds

def pickle_save(obj, path):
	try:
		p_out = open(path,"wb")
		pickle.dump(obj, p_out)
		p_out.close()
	except Exception:
		return "Error in pickle saving"
