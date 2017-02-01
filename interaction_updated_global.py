'''This is the implementation of a tracking by detection framework in which a likelihood map is 
   obtained using the object annotaion in the first and the last frame. Then, based on this map,
   likelihood images for each frame are computed and from that we obtain top 6 windows based on their vote score.
   After this, we compute the optimal path using Dynamic programming. Likelihood maps are based on the paper by Possegger.
   In this, we can manually annotate the object on any frame if we feel that none of the top 6 windows are on the object.'''
 # Size of surroundings has been kept twice the object size
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import math
import copy
import time
import os
from numpy import array
import glob
refPt = []
cropping = False
list_refpt=[]
bin=32 # no. of bins per channel
lamda=0.5 #weight parameter for the combined model
update_para=0.1
lamda_v=0.5
sigma_square=20 # other values can also be chosen 
weight=0.5
lamda_smooth=0.05
num_anno=2
flag=0
# function for labelling object 
def click_and_crop(event, x, y, flags, param):

	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 		
		# draw a rectangle around the region of interest
		cv2.rectangle(img_copy, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", img_copy)
		cv2.waitKey(0)

def mask_bg(object_window,img) :
	''' This function outputs the surrounding pixels
	    Basically, image of background with masked target object'''
	global h_img,w_img
	x,y,w,h=object_window
	h_bg=h*2
	w_bg=2*w
	h_=0.5*h
	w_=0.5*w
	x_bg=int(max(x-(w_),0))
	y_bg=int(max(y-(h_),0))
	x_bg1=int(min(x_bg+w_bg,w_img-1))
	y_bg1=int(min(y_bg+h_bg,h_img-1))
	img[y:y+h,x:x+w]=0
	#print object_window
	#print x_bg,y_bg,x_bg1,y_bg1,img.shape
	bg_img=img[y_bg:y_bg1,x_bg:x_bg1]
	#cv2.imshow("masked_background",bg_img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return bg_img

def label(image) :
	'''This function along with click_and_crop() helps in labelling object and background.
	   Input : Input image
	   Output: selected region of interest(either obj or distractor)'''
	global refPt,cropping,img_copy,clone,list_refpt
	#image1=copy.deepcopy(image)
	#clone = image1.copy()
	#img_copy=image1.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	print "Label the object"
	print "After making a bounding box, press 'c' "
	print "if you wish to select the object again, press 'r' "

	while True:
	# display the image and wait for a keypress
		cv2.imshow("image", img_copy)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()
			img_copy=image.copy()
	 
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break
 	
 	# if there are two reference points, then crop the region of interest
	# from the image and display it
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		cv2.imshow("ROI", roi)
		print "press any key"
		cv2.waitKey(0)

	cv2.destroyAllWindows() # close all open windows
	obj_img=roi  # roi containing the object
	list_refpt.append(refPt)
	print list_refpt,"check_list"
	return obj_img

def prob_obj(hist1,hist2) :
	'''This function creates a look-up table that contains the probability associated with the possible bin values.
	   In our case total bins=32*32*32. This thing will be computed for each frame. Then, when we need to localize the object
	   in the next frame, it will be used.
	   Input: histogram of object,surr/distractors
	   output:array of size 32*32*32 containing probability values for the corresponding bin. This array is called object model.'''
	#hist1=hist1/np.sum(hist1)
	#hist2=hist2/np.sum(hist2)   
	global bin
	delta =.0001
	prob=np.zeros((bin,bin,bin),dtype='float32')
	#likelihood=(np.log((np.maximum(hist1,delta))/np.maximum(hist2,delta)) + 9.22)/18.44 # this is the log likelihood of the object model
	for i in range(bin) :
		for j in range(bin) :
			for k in range(bin) :
				if hist1[i][j][k]>0 or hist2[i][j][k] >0 :
					prob[i][j][k]= hist1[i][j][k]/(hist1[i][j][k]+hist2[i][j][k])
				else :
					prob[i][j][k]=0.5

	return prob

def likelihood_map(prob_map,image) :
	'''This functon generates the likelihood map based on either obj-surr/dist model
	   input: probability map
	   output:likelihood map, an image(each pixel value=corresponding probability)'''

	global h_img,w_img,bin
	sf=256.0/bin
	image_10=image/sf 
	image_10=image_10.astype('uint8')
	# creating a likelihood image acc. to obj-surr or obj-distractor model
	a=image_10[:,:,0]
	a=a.ravel()
	b=image_10[:,:,1]
	b=b.ravel()
	c_=image_10[:,:,2]
	c_=c_.ravel()
	prob_image=prob_map[a,b,c_]
	prob_image=prob_image.reshape((h_img,w_img))
	prob_image1=prob_image*255
	prob_image1=prob_image1.astype('uint8')
	likemap=cv2.applyColorMap(prob_image1, cv2.COLORMAP_JET)
	return likemap,prob_image1

def vote_score(obj_cand,integral_image) :
	#This function computes the sum of probabilities of each pixel in the object window using integral image
	x,y,w,h=obj_cand
	x1=x
	y1=y
	x2=x+w+1
	y2=y+h+1
	score=integral_image[y2][x2]+integral_image[y1][x1]-integral_image[y2][x1]-integral_image[y1][x2]
	return score

def get_distractor(list_score,score_obj,object_window,image) :
	# based on an object window, it computes the distractors windows in a frame that are
	# further used to compute the distractor-aware model  
	# NMS(Non maximal suppression) is also applied
	global lamda_v
	x,y,w,h=object_window
	distractor_mask=np.where(list_score>lamda_v*score_obj) 
	distractor_mask=np.array(distractor_mask)
	distractor_mask[0]=distractor_mask[0]
	distractor_mask[1]=distractor_mask[1]
	dist_img_list=[] # this list will be containing updated distractors
	dist_img_points=[(x,y)]
	for n in range(len(distractor_mask[0])) :
		count_dist=0
		dx=distractor_mask[1][n]
		dy=distractor_mask[0][n]

		for l in range(len(dist_img_points)) :
			diffx=dx-dist_img_points[l][0]
			diffy=dy-dist_img_points[l][1]
			if (diffx>w or diffx<-w or diffy>h or diffy<-h) :   # checking overlapping distractors
				count_dist=count_dist+1
			else :
				w_box=w-math.sqrt(diffx*diffx)	
				h_box=h-math.sqrt(diffy*diffy)
				area=w_box*h_box
				if(area<0.1*w*h) :
					count_dist=count_dist+1
					
		if(count_dist==len(dist_img_points)) :
			distractor=image[dy:dy+h,dx:dx+w]
			dist_img_list.append(distractor)
			dist_img_points.append((dx,dy))
	##################### distractors updated
	return dist_img_list

def top6detections(integral_image) :
	# this function takes in the integral image of the likelihood image and the computes the top 6 windows for a frame
	# NMS is also applied
	global w,h,w_img,h_img
	start_x=0
	start_y=0
	list_score=[]
	list_window=[]
	while(start_x<w_img-w) :
		start_y=0
		while(start_y<h_img-h) :
			obj_cand=(start_x,start_y,w,h)
			score=vote_score(obj_cand,integral_image)
			list_score.append(score)
			list_window.append((start_x,start_y,w,h,score))
			start_y=start_y+5
		start_x=start_x+5	
	sorted_scores=sorted(range(len(list_score)),key=lambda a:list_score[a],reverse=True)

	list_window=np.array(list_window)
	#print list_window[sorted_scores[:10]]
	top6win=list_window[sorted_scores[:]] # selected 6 windows based on score
	final_windows=[top6win[0]]
	win_points=[(top6win[0][0],top6win[0][1])]
	n=1
	while(1) :
		count_win=0
		dx=top6win[n][0]
		dy=top6win[n][1]
		for l in range(len(win_points)) :
			diffx=dx-win_points[l][0]
			diffy=dy-win_points[l][1]
			if (abs(diffx)>w or abs(diffy)>h) :   # checking overlapping distractors
				count_win=count_win+1
			else :
				w_box=w-abs(diffx)	
				h_box=h-abs(diffy)
				area=w_box*h_box
				if(area<0.2*w*h) :
					count_win=count_win+1
					
		if(count_win==len(win_points)) :
			final_windows.append(top6win[n])
			win_points.append((dx,dy))
			if(len(final_windows)==6) :
				break	
		n=n+1	
	return final_windows

def recompute_model(image,obj_img,bg_img,object_window) :
	global bin,hist_comb_obj,hist_comb_bg,hist_comb_dist
	# computing the histograms for object and background
	hist_obj = cv2.calcHist([obj_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
	hist_bg  = cv2.calcHist([bg_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
	# removing the effect of the pixels of the object, as object pixels had (0,0,0) pixel value in bg_img
	hist_bg[0][0][0]=hist_bg[0][0][0]- np.sum(hist_obj)
	hist_comb_obj=hist_comb_obj + hist_obj
	hist_comb_bg=hist_comb_bg + hist_bg
	prob_S=prob_obj(hist_comb_obj,hist_comb_bg)
	color_map_surr,prob_img_surr=likelihood_map(prob_S,image)
	#integral image
	integral_image=cv2.integral(prob_img_surr,sdepth=-1)
	integral_image=integral_image.astype('float32')
	integral_image=integral_image/255

	x,y,w,h=object_window
	score_obj=vote_score(object_window,integral_image) # obj-score
	list_score=np.zeros((h_img-h,w_img-w),dtype='float32')
	
	start_x=0
	start_y=0
	while(start_x<w_img-w) :
		start_y=0
		while(start_y<h_img-h) :
			obj_cand=(start_x,start_y,w,h)
			score=vote_score(obj_cand,integral_image)
			list_score[start_y][start_x]=score
			start_y=start_y+3
		start_x=start_x+3	

	dist_img_list = get_distractor(list_score,score_obj,object_window,image)
	#compute normalized histogram for distractors
	hist_D=np.zeros((bin,bin,bin),dtype='float32')
	for count in range(len(dist_img_list)) :
		hist_dist=cv2.calcHist([dist_img_list[count]],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		hist_D=hist_D+hist_dist
	hist_comb_dist=hist_comb_dist+hist_D
	prob_D=prob_obj(hist_comb_obj,hist_comb_dist)
	prob_comb=prob_S*0.5 + prob_D*0.5
	color_surr,img_surr =likelihood_map(prob_S,image)
	color_dist,img_dist =likelihood_map(prob_D,image)
	color_map,prob_img=likelihood_map(prob_comb,image)
	cv2.imshow("image",color_map1)
	cv2.imshow("original",image)	
	cv2.imshow("obj-surr",color_surr)
	cv2.imshow("obj-dist",color_dist)
	cv2.imshow("like_map",color_map)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return prob_comb,prob_img

if __name__ == "__main__":
	argument=sys.argv
	videoname="dataset_video/girl.mov"
	cap=cv2.VideoCapture(videoname)
	var=0
	if (len(argument)<2) :
		print "\n \n provide an image as input\n\n"
		if var==1 :
			folder_name="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/girl/0001.jpg"
			image=cv2.imread(folder_name +"/00000001.jpg")
			newpath="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/result_trellis"
			if not os.path.exists(newpath) :
				os.makedirs(newpath)
		else :
			#cap.set(cv2.CAP_PROP_POS_FRAMES,)	
			ret,image=cap.read()
			frame_count=cap.get(7)
			num_images=int(frame_count-2)
			image=cv2.resize(image,(500,500)) 

	'''if (len(argument)==2):	
		image=cv2.imread(str(argument[1])) # complete image of the scene

	if var==1 :
		path, dirs, files = os.walk(folder_name).next()
		num_images = len(files)
		filenames=["%04d" % number for number in range(num_images)]
		num_images = len(glob.glob1(folder_name,"*.jpg"))'''		
	
	hist_comb_obj=np.zeros((bin,bin,bin),dtype='float32')
	hist_comb_bg =np.zeros((bin,bin,bin),dtype='float32')
	hist_comb_dist=np.zeros((bin,bin,bin),dtype='float32')
	object_window_list=[]
	images=[]
	# computing the combined histograms from the first and the last frame
	for i in range(num_anno) :
		if i==0 :
			ret,image=cap.read()
			image=cv2.resize(image,(500,500))
			images.append(image)
		else :
			list_refpt=[]
			cap.set(cv2.CAP_PROP_POS_FRAMES,num_images-20)
			ret,image=cap.read()
			image=cv2.resize(image,(500,500))
			images.append(image)
	
		h_img,w_img,c=image.shape
		clone=image.copy()
		img_copy=image.copy()
		obj_img=label(image)  # for labelling object pixels
		h,w,c=obj_img.shape	
		object_window=(list_refpt[0][0][0],list_refpt[0][0][1],w,h)
		x,y,w,h=object_window
		object_window_list.append(object_window)
		frame=copy.deepcopy(image)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow("frame",frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	
		#############
		# creating likelihood maps based on object-surr and object-dist model
		#############
		image1=copy.deepcopy(image)
		bg_img=mask_bg(object_window,image1) # getting background pixels
		# computing the histograms for object and background
		hist_obj = cv2.calcHist([obj_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		hist_bg  = cv2.calcHist([bg_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		# removing the effect of the pixels of the object, as object pixels had (0,0,0) pixel value in bg_img
		hist_bg[0][0][0]=hist_bg[0][0][0]- np.sum(hist_obj)
		#plt.plot(hist_obj[0][0])
		#plt.show() 
		hist_comb_obj=hist_comb_obj + hist_obj
		hist_comb_bg=hist_comb_bg + hist_bg

    # using the combined histogram, creating the likelihood map
    # probability map and likelihood image computation for object-surr model
	prob_S=prob_obj(hist_comb_obj,hist_comb_bg)
	for i in range(num_anno) :
		image=images[i]
		color_map,prob_img_surr=likelihood_map(prob_S,image)
		#integral image
		integral_image=cv2.integral(prob_img_surr,sdepth=-1)
		integral_image=integral_image.astype('float32')
		integral_image=integral_image/255

		x,y,w,h=object_window_list[i]
		score_obj=vote_score(object_window_list[i],integral_image) # obj-score
		list_score=np.zeros((h_img-h,w_img-w),dtype='float32')
		start_x=0
		start_y=0
		while(start_x<w_img-w) :
			start_y=0
			while(start_y<h_img-h) :
				obj_cand=(start_x,start_y,w,h)
				score=vote_score(obj_cand,integral_image)
				list_score[start_y][start_x]=score
				start_y=start_y+3
			start_x=start_x+3	

		dist_img_list = get_distractor(list_score,score_obj,object_window_list[i],image)# computing distractors

		#compute normalized histogram for distractors
		hist_D=np.zeros((bin,bin,bin),dtype='float32')
		for count in range(len(dist_img_list)) :
			hist_dist=cv2.calcHist([dist_img_list[count]],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
			hist_D=hist_D+hist_dist
		hist_comb_dist=hist_comb_dist+hist_D
    # distractor-awareness model
	prob_D=prob_obj(hist_comb_obj,hist_comb_dist)
	prob_comb=prob_S*0.5 + prob_D*0.5
	color_surr,img_surr =likelihood_map(prob_S,images[0])
	color_dist,img_dist =likelihood_map(prob_D,images[0])
	color_map1,prob_img1=likelihood_map(prob_comb,images[0])
	color_map2,prob_img2=likelihood_map(prob_comb,images[1])
	cv2.imshow("image",color_map1)
	cv2.imshow("original",images[0])	
	cv2.imshow("obj-surr",color_surr)
	cv2.imshow("obj-dist",color_dist)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	integral_image=cv2.integral(prob_img1,sdepth=-1)
	integral_image=integral_image.astype('float32')
	integral_image=integral_image/255
	x,y,w,h=object_window_list[0]
	score_obj=vote_score(object_window_list[0],integral_image) # obj-score

	integral_image=cv2.integral(prob_img2,sdepth=-1)
	integral_image=integral_image.astype('float32')
	integral_image=integral_image/255
	score_obj_last=vote_score(object_window_list[1],integral_image)
	M=6
	# Now, using DP, I'll compute the shortest path for tracking
	DP_score=np.zeros((M,num_images),dtype='float32') # matrix containing normalized object candidates' scores
	DP_cost=np.ones((M,num_images),dtype='float32')*float('inf') # initialized to inf, these will contain the final cost associate with each windows
	DP_lastpos=np.zeros((M,num_images),dtype='uint8') # index of the window(0-5) of the window in the previous frame for which cost is minimum to reach that window 
	DP_score[0][0] = score_obj/(h*w)
	DP_score[0][num_images-1]=score_obj_last/(object_window_list[1][2]*object_window_list[1][3])
	DP_cost[0][0]  =-np.log(DP_score[0][0])
	DP_cost[0][num_images-1]=-np.log(DP_score[0][num_images-1])
	node_array=np.zeros((M,4,num_images),dtype='float32') # array containg x,y,w,h of each window
	node_array[0,:,0] = np.array(object_window_list[0])
	node_array[0,:,num_images-1]= np.array(object_window_list[1])
	cap.set(cv2.CAP_PROP_POS_FRAMES,1)
	i=1
	curr_frame_num=i
	enter_count=0
	while(i<num_images) :
		t=time.time()
		ret,image=cap.read()
		if ret==0 :
			break
		image=cv2.resize(image,(500,500))
		color_map,prob_img=likelihood_map(prob_comb,image)
		#cv2.imwrite("results_global_tracking/like/"+str(i)+".jpg",color_map)
		#cv2.imshow('like_map',color_map)
		integral_image=cv2.integral(prob_img,sdepth=-1)
		integral_image=integral_image.astype('float32')
		integral_image=integral_image/255
		top6_detections=top6detections(integral_image) # list containing windows and their scores(x,y,w,h,score)
		top6_detections=np.array(top6_detections)
		#print time.time()-t,"time1"
		# global optimisation using dynamic programming
		if i<num_images-1 :
			node_array[:,0,i],node_array[:,1,i],node_array[:,2,i],node_array[:,3,i]=top6_detections[:,0],top6_detections[:,1],top6_detections[:,2],top6_detections[:,3]
		
		if i==1 :
			for j in range(M) :
				det_term = -math.log(top6_detections[j,4]/(h*w))
				smooth_term= np.square((top6_detections[j,0]-node_array[0,0,0])/60) + np.square((top6_detections[j,1]-node_array[0,1,0])/40)
				DP_cost[j,i] = det_term +lamda_smooth*smooth_term + DP_cost[0,i-1]
				DP_lastpos[j,i]=0

		elif i<num_images-1 :
			for j in range(M):
				det_term=-math.log(top6_detections[j,4]/(h*w))
				smooth_term= np.square((top6_detections[j,0]-node_array[:,0,i-1])/40) + np.square((top6_detections[j,1]-node_array[:,1,i-1])/30)
				cost=det_term + lamda_smooth*smooth_term + DP_cost[:,i-1]
				#print det_term,np.mean(smooth_term*lamda_smooth)
				DP_cost[j,i]= np.amin(cost)	
				DP_lastpos[j,i]= np.argmin(cost)
				#print det_term,smooth_term[DP_lastpos[j,i]]

		else :
			smooth_term= np.square((node_array[0,0,i]-node_array[:,0,i-1])/40) + np.square((node_array[0,1,i]-node_array[:,1,i-1])/30)
			det_term=-math.log(DP_score[0,num_images-1])
			cost=lamda_smooth*smooth_term + DP_cost[:,i-1]
			#print det_term,np.mean(smooth_term*lamda_smooth)
			DP_cost[0,i]= np.amin(cost)	
			DP_lastpos[0,i]= np.argmin(cost)
		font = cv2.FONT_ITALIC
		image_copy=copy.deepcopy(image)
		for count in range(M) :
			a,b,c,d,e=top6_detections[count]
			a=int(a)
			b=int(b)
			c=int(c)
			d=int(d)
			cv2.rectangle(image_copy,(a,b),(a+c,b+d),(0,255,0),1)
			cv2.putText(image_copy,str(count+1),(a,b), font, 0.3,(0,0,255),2,cv2.LINE_AA)
		
		cv2.imshow("image",image_copy)
		cv2.imwrite("results_global_tracking/top6win/"+str(i)+".jpg",image)
		k=cv2.waitKey(10) & 0xff	

		if k== 27 :
			flag=1
			break
		# if anyone wants ro reannotate, press "r" and then this if block will be exexuted
		# for moving the frame forward or backward, press "f" and "b" respectively
		# for annotation press s and the annotate
		if k==114  : # if r is pressed
			cv2.destroyAllWindows()
			while(1) :
				img=np.zeros((400,400),dtype='uint8')
				cv2.putText(img,"if you want to go back then press 'b',",(10,100), font, 0.5,(255,255,255),2,cv2.LINE_AA)
				cv2.putText(img,"for forward press 'f' and to exit press 's'",(10,120), font, 0.5,(255,255,255),2,cv2.LINE_AA)
				cv2.imshow('reannotation',img)
				k=cv2.waitKey(0) & 0xFF
				if k==ord("b") :
					curr_frame_num=int(max(curr_frame_num-5,0))
					print "back",curr_frame_num
					cap.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_num)

				elif k==ord("f") :
					curr_frame_num=int(min(curr_frame_num+5,i))
					print "forward",curr_frame_num
					cap.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_num)
						
				elif(k==ord("s")) :
					print "out"
					break
				if k in {ord("b"),ord("f")} :
					ret,image=cap.read()
					image=cv2.resize(image,(500,500))
					image_rect=copy.deepcopy(image)
					for count in range(M) :
						a,b,c,d = node_array[count,:,curr_frame_num]
						a=int(a)
						b=int(b)
						c=int(c)
						d=int(d)
						cv2.rectangle(image_rect,(a,b),(a+c,b+d),(0,255,0),1)
					cv2.imshow("image_rect",image_rect)
					color_map,prob_img=likelihood_map(prob_comb,image)
					cv2.imshow('like_map',color_map)

			cv2.destroyAllWindows()
			DP_cost[:,curr_frame_num:i+1]=float('inf')
			DP_score[:,curr_frame_num:i+1]=0
			list_refpt=[]
			h_img,w_img,c=image.shape
			clone=image.copy()
			img_copy=image.copy()
			obj_img=label(image)  # for labelling object pixels
			h,w,c=obj_img.shape	
			object_window=(list_refpt[0][0][0],list_refpt[0][0][1],w,h)
			image1=copy.deepcopy(image)
			bg_img=mask_bg(object_window,image1) # getting background pixels
			node_array[0,:,curr_frame_num] = np.array(object_window)
			prob_comb,prob_img=recompute_model(image,obj_img,bg_img,object_window)
			integral_image=cv2.integral(prob_img,sdepth=-1)
			integral_image=integral_image.astype('float32')
			integral_image=integral_image/255
			score_obj=vote_score(object_window,integral_image) # obj-score
			DP_score[0,i]=score_obj/(h*w)
			smooth_term= np.square((node_array[0,0,i]-node_array[:,0,i-1])/40) + np.square((node_array[0,1,i]-node_array[:,1,i-1])/30)
			det_term=-math.log(DP_score[0,i])
			cost=lamda_smooth*smooth_term + DP_cost[:,i-1]
			#print det_term,np.mean(smooth_term*lamda_smooth)
			DP_cost[0,i]= np.amin(cost)	
			DP_lastpos[0,i]= np.argmin(cost)
			enter_count=i
			i=curr_frame_num
		print 1/(time.time()-t),"fps"
		i=i+1
		print i,"see i"
		curr_frame_num=i

	# back-tracking to compute the optimal path
	list_index=[0]

	for i in range(int(num_images)-1) :
		lastpos=DP_lastpos[list_index[i]][num_images-1-i]
		list_index.append(lastpos)

	templist=list_index.reverse()
	list_index=list_index[::]
	print "length",len(list_index)
	cap=cv2.VideoCapture(videoname)
	for i in range(int(num_images)) :
		ret,image=cap.read()
		if image is None or flag==1:
			break
		image=cv2.resize(image,(500,500))
		a,b,c,d = node_array[list_index[i],:,i]
		a=int(a)
		b=int(b)
		c=int(c)
		d=int(d)
		cv2.rectangle(image,(a,b),(a+c,b+d),(0,255,0),1)
		cv2.imshow("image",image)
		cv2.imwrite("results_global_tracking/images/"+str(i)+".jpg",image)
		cv2.waitKey(10)



		




		
