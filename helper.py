# helper.py
import pickle
import cv2
path = "kp.pickle"
path1 = "ds.pickle"

#-- Unpickling data
def unpickle(path):
	import pickle
	pickle_in = open(path, 'rb')
	data = pickle.load(pickle_in, encoding='bytes')
	return data

# Working if change in temp 
def keypoint_load(path):
	kp = []
	pickle_in = open(path, 'rb')
	data = pickle.load(pickle_in, encoding='bytes')
	for d in data:
		temp_kp = []
		for p in d:
			temp = (cv2.KeyPoint(x=p[0][0],y=p[0][0],_size=p[1],_angle=p[2],_response=p[3],_octave=p[4],_class_id = p[5]))
			temp_kp.append(temp)
		kp.append(temp_kp)
	return kp

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
		return -1

def keypoint_save(keypoints,path):
	arr =[] 
	for point in keypoints:
		temp_kp = []
		for kp in point:
			temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
			temp_kp.append(temp)
		arr.append(temp_kp)
	print(pickle_save(arr, path))
