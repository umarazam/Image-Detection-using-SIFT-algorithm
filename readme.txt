# Imporvement:

1: Pickle images descriptors and kps(if it does not create issue);

2: reshape_1D returns the numpy_ndarray
-->
2: Get the maximum shape descriptor from list  --> Convert all other to that

3: --> End of the day: - BFMatcher improve it.
		Try try.

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


for k,v in d.items():
...     if v > 13:
...             d.pop(k,'Not Found')
...             temp = { 15:13}
...             d.update(temp)
...             del temp



#match = bf_matcher(ds,ds_t[1],3)
	#match = match.sort()	


