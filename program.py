import cv2
import numpy as np
from math import sqrt
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
	i=0
	for im in img:
		if i> 3:
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
	length = len(n_des)
	for a in descriptors:
		length_1 = len(a)
		pass

def reshape(ds1, ds2):
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
		print('Here')
		dif = len(ds2)-len(ds1) # --> Correct
		zeros = np.zeros((1, dif))
		ds1= np.append(ds1, zeros)
		print(len(ds1))
		print(len(ds2))

def kp_matching(kp1,kp2):
	if kp1 == kp2:
		return True
	return False

def knn_intialize(data,des):
	knn = cv2.ml.KNearest_create()
	print(len(data))
	print(len(des))
	for im,ds in zip(data,des):
		knn.train(im,ds)
	return knn

def knn_result(knn, k ,data):
	ret,results,neighbours, dis = knn.find_nearest(data,k)
	return ret, results, neightbours,dist


def main():
	forest_images_path = '/home/umar/dip_project/New_Try/Database/forest/*.jpg'
	images = load_images(forest_images_path)

	print("SIFT Working........")
	kps, ds = featureExtraction(images)
#	Classification

	reshape(ds[1],ds[0]) # --> Reshaped correctly


if __name__ == '__main__':
	main()
