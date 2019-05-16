import cv2
import numpy as np
from math import sqrt, floor
from glob import glob
from timeit import default_timer as timer
from helper import *

def load_images(path):
	try:
		filenames = glob(path)
		filenames.sort()
		images= [cv2.imread(img) for img in filenames]
		return images
	except Exception:
		print(f"Couldn't Load the images with path {path}")

# SIFT Implementation
def featureExtraction(img):
	import numpy as np
	kps = []
	ds =[]
	sift = cv2.xfeatures2d.SIFT_create()
	for im in img:
		kp,d = sift.detectAndCompute(im,None)
		ds.append(d)
		kps.append(kp)
	return kps,ds

def euclidean_Distance(x, x1):
	res = sqrt((x-x1) **2)
	return res

def knn(k, descriptors, n_des):
	matches = {}
	for index,a in enumerate(descriptors):
		d_train, d_test = reshape_1D(a, n_des)
		res=floor(sqrt(sum((d_train-d_test) **2)))
		temp_d = {index:res}
		print(f'index: {index}, Result: {res}')
		if len(matches) < k:
			matches.update(temp_d)
		else:
			for key,v in matches.items():
				if v > res:
					matches.pop(key,'Not Found')
					print('Updated')
					matches.update(temp_d)
					break
		del temp_d
	print(matches)
	return matches
# Got result --check

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

def kp_matching(kp1,kp2):
	if kp1 == kp2:
		return True
	return False

def main():
	forest_images_path = '/home/umar/dip_project/New_Try/Database/forest/*.jpg'
	test_images_path = '/home/umar/dip_project/New_Try/Database/test/*.jpg'
	images = load_images(forest_images_path)

	images_test = load_images(test_images_path)

	print("SIFT Working for training........")
	kps, ds = featureExtraction(images)
	print("SIFT Working for testin........")
	kps_t, ds_t = featureExtraction(images_test)

#	Classification

	#t,te = reshape_1D(ds[0],ds_t[0]) # --> Reshaped correctly
	matches = knn(3, ds, ds_t[1])
	print("Matches Images")
	index = matches.keys()

	pridicted_index = []

	for a in index:
		pridicted_index.append(a)

	print(pridicted_index[0])
	print("Original Image")
	showImg(images_test[1],1)
	print("Pridicted Image")
	showImg(images[pridicted_index[0]],1)
	showImg(images[pridicted_index[1]],1)
	showImg(images[pridicted_index[2]],1)

if __name__ == '__main__':
	main()
