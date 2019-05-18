#program.py

import cv2
import numpy as np
from math import sqrt, floor
from timeit import default_timer as timer
from helper import *

# --> Names for pickle objects
path_kp_train = "kp_train.pickle"
#path_kp_test = "kp_test.pickle"

path_ds_train = "ds_train.pickle"
#path_ds_test = "ds_test.pickle"

# SIFT Implementation
def featureExtraction(img):
	import numpy as np
	kps = []
	ds =[]
#	sift = cv2.xfeatures2d.SIFT_create()
	sift = cv2.xfeatures2d.SIFT_create(0,3,0.04,10,1.8)
	for im in img:
		kp,d = sift.detectAndCompute(im,None)
		ds.append(d)
		kps.append(kp)
	return kps,ds

def knn(k, descriptors, n_des):
	matches = {}
	for index,a in enumerate(descriptors):
		d_train, d_test = reshape_1D(a, n_des)
		res=floor(sqrt(sum((d_train-d_test) **2)))
		temp_d = {index:res}
	#	print(f'index: {index}, Result: {res}')
		if len(matches) < k:
			matches.update(temp_d)
		else:
			for key,v in matches.items():
				if v > res:
					matches.pop(key,'Not Found')
					#print('Updated')
					matches.update(temp_d)
					break
		del temp_d 
	print(matches)
	return matches

def reshape_1D(ds1, ds2):
	# Reshape array function
	flag = True
	ds1 = ds1.flatten()
	ds2 = ds2.flatten()
	if len(ds2) > len(ds1):
		flag = False
	if flag:  #	It will reshape and 1-D ds2 will be of same shape ds1
		dif = len(ds1)-len(ds2) # --> Correct
		zeros = np.zeros((1, dif))
		ds2= np.append(ds2, zeros)
	else:
		dif = len(ds2)-len(ds1) # --> Correct
		zeros = np.zeros((1, dif))
		ds1= np.append(ds1, zeros)
	
	return ds1, ds2

# BFMatcher implementation:

def bf_matcher(des, n_des,k): # --> Useless
	bf = cv2.BFMatcher()
	matches = []
	for i,a in enumerate(des):
		match = bf.knnMatch(a,n_des,3)
		li = [i,match]
		matches.append(li)
	print(matches)
	return matches

def main():
	forest_images_path = '/home/umar/dip_project/new_try/Database/Dataset1/Raw_Data/tallbuilding/*.jpg'
	test_images_path = '/home/umar/dip_project/new_try/Database/Dataset1/test/*.jpg'
	images = load_images(forest_images_path)
	images_test = load_images(test_images_path)

	try:
		kps = keypoint_load(path_kp_train)
		ds = pickle_load(path_ds_train)
	#	kps_t = keypoint_load(path_kp_test)
	#	ds_t = pickle_load(path_ds_test)
		print('Data loaded successfully...!')
	except Exception:
		print('Generating keypoints and descriptors')
		print("SIFT Working for training data.")
		ts = timer()
		kps, ds = featureExtraction(images)
		tf = timer()
		(keypoint_save(kps,path_kp_train))
		(pickle_save(ds,path_ds_train))
	#	(keypoint_save(kps_t,path_kp_test))
	#	(pickle_save(ds_t,path_ds_test))
		print(f'Total time took for generating keypoints and descriptors {tf-ts}')

	print("SIFT Working for testing data.")
	kps_t, ds_t = featureExtraction(images_test)
	print("Done")
	print("Starting classification")
#	Classification 
	ts = timer()
	matches = knn(3, ds, ds_t[0])
	tf = timer()
	print(f'Time took to classify images {tf-ts}')
	print("Matches Images")
	index = matches.keys()
	images_window = []
	images_window.append(images_test[0])
	for a in index:
		images_window.append(images[a])

	# 1st--> Test image other's pridected images

	disImage(images_window)
	print("Pridicted Image")
	#showImg(images_test[0],1)
if __name__ == '__main__':
	main()
