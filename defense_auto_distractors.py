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
bin=10 # no. of bins per channel
lamda=0.5 #weight parameter for the combined model
update_para=0.1
lamda_v=0.5
sigma_square=1 # other values can also be chosen  

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
	x_bg=max(x-(w_),0)
	y_bg=max(y-(h_),0)
	x_bg1=min(x_bg+w_bg,w_img-1)
	y_bg1=min(y_bg+h_bg,h_img-1)
	img[y:y+h,x:x+w]=0
	#print object_window
	#print x_bg,y_bg,x_bg1,y_bg1,img.shape
	bg_img=img[y_bg:y_bg1,x_bg:x_bg1]
	#cv2.imshow("masked_background",bg_img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return bg_img

def likelihood_map((hist1,hist2),image) :
	'''This functon generates the likelihood map based on either obj-surr/dist model
	   input: histogram of object,surr/distractors and input image
	   output:likelihood map, an image(each pixel value=corresponding probability)'''
	global h_img,w_img
	H=hist1+hist2 # histogram of I(O U S) or I(O U D)U is union 
	image_10=image/25.6 # histogram has 10 bins
	image_10=image_10.astype('uint8')
	# creating a likelihood image acc. to obj-surr or obj-distractor model
	a=image_10[:,:,0]
	a=a.ravel()
	b=image_10[:,:,1]
	b=b.ravel()
	c_=image_10[:,:,2]
	c_=c_.ravel()
	H_obj=hist1[a,b,c_] # image with pixel value=bin count of the pixel value at the same location in original image
	H_img=H[a,b,c_]
	Prob1=np.zeros((h_img*w_img,),dtype='float')
	H_obj=H_obj.astype('float')
	H_img=H_img.astype('float')
	mask=H_img==0
	#print mask,"check itjhjnkjkjkjk"
	Prob1[~mask]=np.divide(H_obj[~mask],H_img[~mask])
	Prob1[mask]=0.5
	Prob1=Prob1.reshape((h_img,w_img))
	Prob2=(Prob1)*255
	Prob2=Prob2.astype('uint8')
	likemap=cv2.applyColorMap(Prob2, cv2.COLORMAP_JET)
	return likemap,Prob1

def prob_obj(hist1,hist2) :
	'''This function creates a look-up table that contains the probability associated with the possible bin values.
	   In our case total bins=10*10*10. This thing will be computed for each frame. Then, when we need to localize the object
	   in the next frame, it will be used.
	   Input: histogram of object,surr/distractors
	   output:array of size 10x10x10 containing probability values for the corresponding bin. This array is called object model.'''
	prob=np.zeros((10,10,10),dtype='float32')

	for i in range(10) :
		for j in range(10) :
			for k in range(10) :
				if hist1[i][j][k]>0 or hist2[i][j][k] >0 :
					prob[i][j][k]= hist1[i][j][k]/(hist1[i][j][k]+hist2[i][j][k])
				else :
					prob[i][j][k]=0.5

	return prob

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

def update_model(Prob_comb,Prob_comb_new) :
	'''This function is used to update the object model to adapt to the changing conditions.
	   Input : current_object model; object model computed using prob_obj() for the current frame
	   Output: new_object model'''
	global update_para
	Prob_comb=Prob_comb_new*update_para + (1-update_para)*Prob_comb
	return Prob_comb

def vote_score(prob_comb,obj_candidate) :
	'''vote score is computed based on the current object model to localise the object in the search region
	    Input : object_model(combined probability),object candidate
	    Output: vote_score'''
	image_10=obj_candidate/25.6 # histogram has 10 bins
	image_10=image_10.astype('uint8')
	# creating a likelihood image acc. to obj-surr or obj-distractor model
	a=image_10[:,:,0]
	a=a.ravel()
	b=image_10[:,:,1]
	b=b.ravel()
	c_=image_10[:,:,2]
	c_=c_.ravel()
	prob_candidate=prob_comb[a,b,c_]
	score=np.sum(prob_candidate)
	return score

def distance_score(cand_grid,c_x,c_y) :
	'''This is computed so as to penalize the large object movements in the successive frames
	   Input: Array containing pixel locations of the current candidate,center of current_object_window'''
	cand_grid_x=cand_grid[1]
	cand_grid_y=cand_grid[0]
	cand_grid_x=cand_grid_x-c_x
	cand_grid_y=cand_grid_y-c_y
	cand_grid_x=np.square(cand_grid_x)
	cand_grid_y=np.square(cand_grid_y)
	cand_grid_added=(cand_grid_x+cand_grid_y)*(-1/(2*sigma_square))
	exp=np.exp(cand_grid_added)
	score=np.sum(exp)
	
	return score
def obj_cand_new6(object_window,new_img,prob_comb,number) :
	global h_img,w_img
	list_obj_window=[]
	list_window_scores=[]
	new_frame=copy.deepcopy(new_img)
	x,y,w,h=object_window 

	# setting variables for search window.. Also putting constraints like
	# variables shouldn't go beyond width/height of image

	# leftmost and topmost corner
	start_x=int(max(0,x-w)) 
	start_y=int(max(0,y-h))
	# rightmost and bottommost corner
	end_x=int(min(x+w,w_img-w))
	end_y=int(min(y+h,h_img-h))
	# center of object-window
	c_x=int(x+(w/2)) 
	c_y=int(y+(h/2))
	y0=start_y
	x0=start_x
	list_score=np.zeros((end_y-y0,end_x-x0),dtype='float') # array for containing vote-scores of all the candidates, will be used for updating distractors
	# score for all the candidates is computed and simultaneously filing the above array
	# Also, we calculate the obj-cand with max score 
	while (start_x<end_x) :

		start_y=int(max(y-h,0))
		while(start_y<end_y) :
			#cand_grid=np.mgrid[start_y:start_y+h,start_x:start_x+w]
			obj_cand =new_img[start_y:start_y+h,start_x:start_x+w,:]
			#score1=distance_score(cand_grid,c_x,c_y) # distance-score for the combined object candidate
			score=vote_score(prob_comb,obj_cand)  # vote-score for the combined object candidate
			list_score[start_y-y0][start_x-x0]=score	
			window=(start_x,start_y,w,h)
			list_obj_window.append((window,score))
			list_window_scores.append(score)
			start_y=start_y+5 # sampling after 5 pixels
		start_x=start_x+5	

	list_window_scores=np.array(list_window_scores)
	list_obj_window=np.array(list_obj_window)
	sorted_scores=sorted(range(len(list_window_scores)),key=lambda x:list_window_scores[x],reverse=True)
	top6_index= sorted_scores[:number]
	top6_window=list_obj_window[top6_index]
	counter=0
	prob_comb6=[]
	while(counter<number) :
		obj_window=top6_window[counter][0]
		point1=(obj_window[0],obj_window[1])
		point2=(point1[0]+w,point1[1]+h) # diagonally opposite point to point1
		obj_img=new_frame[point1[1]:point2[1],point1[0]:point2[0]] # updated object roi
		new_frame1=copy.deepcopy(new_frame) 
		bg_img =mask_bg(object_window,new_frame1) # updated surrounding image
		hist_obj = cv2.calcHist([obj_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		hist_bg  = cv2.calcHist([bg_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		# removing the effect of pixels having value (0,0,0) from bg which were present in object-region
		hist_bg[0][0][0]=hist_bg[0][0][0]- np.sum(hist_obj)
		# probabilities of ob-surr
		prob_S=prob_obj(hist_obj,hist_bg)
		prob_comb=update_model(prob_comb,prob_S)
		prob_comb6.append(prob_comb)
		counter=counter+1

	return top6_window,prob_comb6

def min_func(alpha,i,obj_window) :
	global x_y_list,DP_cost,DP_score,DP_lastpos
	x2,y2,w,h=obj_window
	k=0
	cost_min=float('inf')
	while(k<6) :
		x1,y1,w,h=x_y_list[k][i-1]
		distance_wt=np.exp(-(math.pow(x1-x2,2)+math.pow(y1-y2,2)))
		cost=-math.log(DP_score[alpha][i]+distance_wt) + DP_cost[k][i-1]
		if(cost<cost_min) :
			cost_min=cost
			box_no=k
		k=k+1	
	DP_cost[alpha][i]=cost_min
	DP_lastpos[alpha][i]=box_no+1

if __name__ == "__main__":
	argument=sys.argv
	cap=cv2.VideoCapture("theater.mp4")
	var=1
	if (len(argument)<2) :
		print "\n \n provide an image as input\n\n"
		if var==1 :
			folder_name="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/vot14/car"
			image=cv2.imread(folder_name +"/00000001.jpg")
			newpath="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/result_trellis"
			if not os.path.exists(newpath) :
				os.makedirs(newpath)
		else :	
			ret,image=cap.read()
			image=cv2.resize(image,(500,500)) 

	if (len(argument)==2):	
		image=cv2.imread(str(argument[1])) # complete image of the scene

	if var==1 :
		path, dirs, files = os.walk(folder_name).next()
		num_images = len(files)
		filenames=["%08d" % number for number in range(num_images)]
	num_images = len(glob.glob1(folder_name,"*.jpg"))	
	prob_anno=[]
	Prob_anno=[]
	object_window_list=[]
	color_maps=[]
	print num_images,"num_images"
	for i in range(2) :
		
		if i==1 :
			image=cv2.imread(folder_name+"/"+filenames[num_images-1]+".jpg")
			list_refpt=[]
		h_img,w_img,c=image.shape
		clone=image.copy()
		img_copy=image.copy()
		obj_img=label(image)  # for labelling object pixels
		dist_img_list=[]
		frame=image
		cv2.destroyAllWindows()
		cv2.imshow("frame",frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	
		#############
		# creating likelihood maps based on object-surr and object-dist model
		#############
		h,w,c=obj_img.shape	
		image1=copy.deepcopy(image)
		object_window=(list_refpt[0][0][0],list_refpt[0][0][1],w,h)
		bg_img=mask_bg(object_window,image1) # getting background pixels
		# computing the histograms for object and background
		hist_obj = cv2.calcHist([obj_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		hist_bg  = cv2.calcHist([bg_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		x,y=(list_refpt[0][0][0],list_refpt[0][0][1])
		object_window_list.append(object_window)
		obj_cand=image[y:y+h,x:x+w]
		prob_S=prob_obj(hist_obj,hist_bg)
		score_obj=vote_score(prob_S,obj_cand) # in this case obj_cand is the annotated object in the last frame itself
		#looking for distractors

		# setting variables for search window.. Also putting constraints like
		# variables shouldn't go beyond width/height of image

		# leftmost and topmost corner
		start_x=int(max(0,x-w)) 
		start_y=int(max(0,y-h))
		# rightmost and bottommost corner
		end_x=int(min(x+w,w_img-w))
		end_y=int(min(y+h,h_img-h))
		# center of object-window
		c_x=int(x+(w/2)) 
		c_y=int(y+(h/2))
		score_max=0
		y0=start_y
		x0=start_x
		list_score=np.zeros((end_y-y0,end_x-x0),dtype='float') # array for containing vote-scores of all the candidates, will be used for updating distractors
		# score for all the candidates is computed and simultaneously filing the above array
		# Also, we calculate the obj-cand with max score 
		while (start_x<end_x) :

			start_y=int(max(y-h,0))
			while(start_y<end_y) :
				obj_cand =image[start_y:start_y+h,start_x:start_x+w,:]
				score=vote_score(prob_S,obj_cand)  # vote-score for the combined object candidate
				list_score[start_y-y0][start_x-x0]=score
				start_y=start_y+5 # sampling after 5 pixels
			start_x=start_x+5	

		new_frame1=copy.deepcopy(image) 
		# checking the condition for an object cand to be a distractor
		#####
		distractor_mask=np.where(list_score>lamda_v*score_obj) 
		distractor_mask=np.array(distractor_mask)
		distractor_mask[0]=distractor_mask[0]+y0
		distractor_mask[1]=distractor_mask[1]+x0
		dist_img_list=[] # this list will be containing updated distractors
		dist_img_points=[(x,y)]
		for n in range(len(distractor_mask[0])) :
			count_dist=0
			dx=distractor_mask[1][n]
			dy=distractor_mask[0][n]

			for l in range(len(dist_img_points)) :
				diffx=dx-dist_img_points[l][0]
				diffy=dy-dist_img_points[l][1]
				if (diffx>w or diffx<-w or diffy>h or diffy<-h) :
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

		print object_window
		hist_D=np.empty_like(hist_obj)
		for i in range(len(dist_img_list)) :
			print "i",i
			hist_dist=cv2.calcHist([dist_img_list[i]],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
			hist_D=hist_D+hist_dist

		# removing the effect of the pixels of the object, as object pixels had (0,0,0) pixel value in bg_img
		hist_bg[0][0][0]=hist_bg[0][0][0]- np.sum(hist_obj) 
		color_map_surr,Prob_surr=likelihood_map((hist_obj,hist_bg),image) # call likelihood function for obj-surr model
		color_map_dist,Prob_dist=likelihood_map((hist_obj,hist_D),image)# call likelihood function for obj-dist model

		#final_map = color_map_dist.astype('float32')*lamda + color_map_surr.astype('float32')*(1-lamda)
		Prob_comb=Prob_surr*(1-lamda)+Prob_dist*lamda
		Prob_anno.append(Prob_comb)
		final_map=Prob_comb*255
		color_map_final=final_map.astype('uint8')
		color_map_final=cv2.applyColorMap(color_map_final, cv2.COLORMAP_JET)
		color_maps.append(color_map_final)
		cv2.imshow("obj-surr model",color_map_surr)
		cv2.imshow("distractor-aware model",color_map_dist)
		cv2.imshow("combined",color_map_final)
		cv2.imshow("original_frame",frame)
		cv2.waitKey(0)	
		cv2.destroyAllWindows()
		
		#prob as per bin no.
		prob_S=prob_obj(hist_obj,hist_bg)
		prob_D=prob_obj(hist_obj,hist_D)
		prob_comb=prob_D*lamda+prob_S*(1-lamda)
		prob_anno.append(prob_comb)



	prob_comb=(prob_anno[0]+prob_anno[1])/2
	image=cv2.imread(folder_name +"/00000001.jpg")
	image_10=image/25.6 # histogram has 10 bins
	image_10=image_10.astype('uint8')
	# creating a likelihood image acc. to combined model
	a=image_10[:,:,0]
	a=a.ravel()
	b=image_10[:,:,1]
	b=b.ravel()
	c_=image_10[:,:,2]
	c_=c_.ravel()
	prob_image=prob_comb[a,b,c_]
	prob_image=prob_image.reshape((h_img,w_img))
	prob_image=prob_image*255
	prob_image=prob_image.astype('uint8')
	color_map_final=cv2.applyColorMap(prob_image, cv2.COLORMAP_JET)

	cv2.imshow("combined colormap",color_map_final)
	cv2.imshow("colormap",color_maps[0])
	cv2.imshow("image",image)
	cv2.waitKey(0)	
	cv2.destroyAllWindows()	
	
	object_window=object_window_list[0]
	DP_score=np.zeros((6,num_images),dtype='float32')
	DP_cost=np.zeros((6,num_images),dtype='float32')
	DP_lastpos=np.zeros((6,num_images),dtype='uint8')
	#for last frame
	object_window=object_window_list[1]
	x,y,w,h=object_window
	obj_cand=image[y:y+h,x:x+w]
	score=vote_score(prob_comb,obj_cand) # in this case obj_cand is the annotated object in the last frame itself
	DP_score[0][num_images-1]=score/(h*w)

	# for the very first frame, filling in DP array
	object_window=object_window_list[0]
	x,y,w,h=object_window
	obj_cand=image[y:y+h,x:x+w]
	score=vote_score(prob_comb,obj_cand) # in this case obj_cand is the annotated object in the first frame itself
	DP_score[0][0]=score/(h*w) # other rows are of no use to us for the first frame as there is only one obj_cand here
	DP_lastpos[0][0]=1
	DP_cost[0][0]=-math.log(score/(h*w)) # normalized score has been used
	obj_prev_window=[object_window]
	prob_comb6_prev=[prob_comb]
	x_y_list=[[object_window],[object_window],[object_window],[object_window],[object_window],[object_window]]
	i=1
	while(i<num_images-1) :
	
		t=time.time()
		# var variable is 1 when reading from a image files..if reading from a video set var=0 
		if var==1 :
			new_img=cv2.imread(folder_name+"/"+filenames[i]+".jpg")
		else :	
			ret,new_img=cap.read()
			new_img=cv2.resize(new_img,(500,500))

		if(new_img==None) :
			break
		count=0
		current_windows36=[]
		current_prob36=[]
		scores36=[]		
		while(count<6 and DP_lastpos[count][i-1]!=0) :
			if(i==1) :
				number=6
			else :
				number=1	
			#print "count",count, prob_comb6_prev[count].shape,len(prob_comb6_prev)	
			list_obj_window,prob_comb6=obj_cand_new6(obj_prev_window[count],new_img,prob_comb6_prev[count],number)
			#print len(prob_comb6),prob_comb6[0].shape,"prob shapes"
			current_windows36.append(list_obj_window)
			current_prob36.append(prob_comb6)
			scores36.append([score[1] for score in list_obj_window])
			count=count+1
		#print len(list_obj_window),len(prob_comb6),len(current_windows36)
		#print current_windows36	
			
		scores36=np.array(scores36)
		scores36=scores36.ravel()
		alpha=0
		while(alpha<6) :
			DP_score[alpha][i]=scores36[alpha]/(h*w)
			if i==1 :
				DP_lastpos[alpha][i]=1
				x1,y1,h,w=obj_prev_window[0]
				x2,y2,h,w=current_windows36[0][alpha][0]
				pos=DP_lastpos[alpha][i]-1
				distance_wt= np.exp(-(math.pow(x1-x2,2)+math.pow(y1-y2,2)))
				DP_cost[alpha][i] = -math.log(DP_score[alpha][i] + distance_wt) +DP_cost[pos][i-1]
			else :
				# here we will need to minimise the functon over k
				min_func(alpha,i,current_windows36[alpha][0][0])
					
			alpha=alpha+1	

		obj_prev_window=[]
		prob_comb6_prev=[]
		alpha=0
		while(alpha<6) :
			if(i==1) :
				x_y_list[alpha].append(current_windows36[0][alpha][0])
				obj_prev_window.append(current_windows36[0][alpha][0])
				prob_comb6_prev.append(current_prob36[0][alpha])
			
			else :
				x_y_list[alpha].append(current_windows36[alpha][0][0])	
				obj_prev_window.append(current_windows36[alpha][0][0])
				prob_comb6_prev.append(current_prob36[alpha][0])
			alpha=alpha+1	


		cv2.imshow("new_img",new_img)
		k=cv2.waitKey(10) & 0xff
		if k==27 :
			break
		i=i+1

	# for final frame
	object_window=object_window_list[1]
	min_func(0,num_images-1,object_window)
	min_path=[object_window]
	count=num_images-1
	row=0
	while(count>0) :
		print "hello"
		row=DP_lastpos[row][count]-1
		count=count-1
		obj_win=x_y_list[row][count]
		min_path.append(obj_win)
		print min_path
	min_path1=min_path.reverse()
	min_path1= min_path[::1]
	print min_path1
	for i in range(1,num_images+1) :
		print i,filenames[i]
		new_img=cv2.imread(folder_name+"/"+filenames[i+1]+".jpg")
		if new_img==None :
			break
		x,y,w,h=min_path[i-1]
		cv2.rectangle(new_img,(x,y),(x+w,y+h),(0, 255, 0), 2) 
		cv2.imshow("new",new_img)
		cv2.waitKey(100)
		
	#print np.argmin(DP_cost[:,num_images-1])
	cv2.destroyAllWindows()