import cv2
from math import sqrt
from glob import glob
from timeit import default_timer as timer
from helper import *

def load_images(path):
	try:
		filenames = glob(path)
		#filenames.sort()
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
	i=0
	for im in img:
		if i > 2:
			break
		i+=1
		kp,d = sift.detectAndCompute(im,None)
		ds.append(d)
		kps.append(kp)

	return kps,ds

def euclidean_Distance(x=1,y=1, x1=0, y1 =0):
	res = sqrt((x-x1) **2 + (y-y1) **2)
	return res

def knn(k, descriptors, n_des):
	dis_list = []
	for a in descriptors:
		pass

def kp_matching(kp1,kp2):
	if kp1 == kp2:
		return True
	return False

def main():
	forest_images_path = '/home/umar/dip_project/New_Try/Database/forest/*.jpg'
	images = load_images(forest_images_path)

	print("SIFT Working........")
	kps, ds = featureExtraction(images)
	print(kp_matching(kps[0],kps[1]))


if __name__ == '__main__':
	main()

