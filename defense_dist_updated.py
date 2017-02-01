'''In this, I have tried to use 32 bins for each channel. The tasks this code does are:
   1. computing model based on first and last frame.
   2. In this I update the distractors by using a sliding-window on the entire 
   	  likelihood image(obtained after combined model using search-window=3xobject-window). This is done to supresss all the distractors.
   3. Then, use this model to predict location in this subsequent frames.
   	  Along with this, I have used dynamic programming to compute the min cost path.'''

 # Size of surroundings has been kept twice the object size
import cv2
import numpy as np
import sys
from matplotlib import pyplot as pltc
import math
import copy
import time
import os
from numpy import array
import glob
refPt = []
cropping = False
list_refpt=[]
bin=16 # no. of bins per channel
lamda=0.5 #weight parameter for the combined model
update_para=0.1
lamda_v=0.5
sigma_square=20 # other values can also be chosen 
weight=0.5


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

def update_model(Prob_comb,Prob_comb_new) :
	'''This function is used to update the object model to adapt to the changing conditions.
	   Input : current_object model; object model computed using prob_obj() for the current frame
	   Output: new_object model'''
	global update_para
	Prob_comb=Prob_comb_new*update_para + (1-update_para)*Prob_comb
	return Prob_comb

def vote_score(obj_cand,integral_image) :
	x,y,w,h=obj_cand
	x1=x
	y1=y
	x2=x+w+1
	y2=y+h+1
	score=integral_image[y2][x2]+integral_image[y1][x1]-integral_image[y2][x1]-integral_image[y1][x2]
	return score

def summed_table(start,end,center,object_window) :
	start_x,start_y=start
	end_x,end_y=end
	c_x,c_y=center
	x,y,w,h=object_window
	search_window=np.mgrid[start_y:end_y+h,start_x:end_x+w]
	search_window[0]= np.square(search_window[0]-c_y)
	search_window[1]= np.square(search_window[1]-c_x)
	square_window   = -(search_window[0]+search_window[1])/(sigma_square)
	exp= np.exp(square_window/sigma_square)
	table=exp.cumsum(axis=0).cumsum(axis=1)
	return table

def obj_cand_new6(object_window,new_img,prob_comb,number) :
	global h_img,w_img,lamda_v
	list_obj_window=[]
	list_window_scores=[]
	new_frame=copy.deepcopy(new_img)
	x,y,w,h=object_window 
	color_map_surr,prob_img_surr=likelihood_map(prob_comb,new_img) # call likelihood function for obj-surr model
	integral_image=cv2.integral(prob_img_surr,sdepth=-1)
	integral_image=integral_image.astype('float32')
	integral_image=integral_image/255
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
	
	h_3=3*h
	sum_table=summed_table((start_x,start_y),(end_x,end_y),(c_x,c_y),object_window)
	sum_h,sum_w= sum_table.shape
	sum_table=np.concatenate((np.zeros((1,sum_w),dtype='float64'),sum_table),axis=0)
	sum_table=np.concatenate((np.zeros((sum_h+1,1),dtype='float64'),sum_table),axis=1)
	list_score=np.zeros((end_y-y0,end_x-x0),dtype='float') # array for containing vote-scores of all the candidates, will be used for updating distractors
	# score for all the candidates is computed and simultaneously filing the above array
	# Also, we calculate the obj-cand with max score 
	while (start_x<end_x) :

		start_y=int(max(y-h,0))
		while(start_y<end_y) :
			#cand_grid=np.mgrid[start_y:start_y+h,start_x:start_x+w]
			obj_cand=(start_x,start_y,w,h)
			score_obj=vote_score(obj_cand,integral_image)
			a1=start_x-x0
			b1=start_y-y0
			a2=a1+w+1
			b2=b1+h+1
			#print a1,b1,a2,b2,"index"
			distance_score_obj=sum_table[b2][a2]+sum_table[b1][a1]-sum_table[b2][a1]-sum_table[b1][a2]
			distance_score_obj=distance_score_obj # normalization
			final_score=score_obj*distance_score_obj
			#print distance_score_obj,sum_table[b2][a2],sum_table[b1][a1],sum_table[b2][a1],sum_table[b1][a2],"distance scores"
			if distance_score_obj<0 :
				print "horrible"
			list_score[start_y-y0][start_x-x0]=	score_obj
			window=(start_x,start_y,w,h)
			#print "windows",window,final_score
			list_obj_window.append((window,final_score))
			list_window_scores.append(final_score)
			start_y=start_y+5 # sampling after 5 pixels
		start_x=start_x+5	

	list_window_scores=np.array(list_window_scores)
	list_obj_window=np.array(list_obj_window)
	sorted_scores=sorted(range(len(list_window_scores)),key=lambda x:list_window_scores[x],reverse=True)
	top6_index= sorted_scores[:number]
	#print top6_index,"top6"
	top6_window=list_obj_window[top6_index]
	counter=0
	prob_comb6=[]
	while(counter<number) :
		obj_window=top6_window[counter][0]
		x,y,w,h=obj_window
		score_obj=vote_score(obj_window,integral_image)
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
		# checking the condition for an object cand to be a distractor
		#####
		dist_img_list=get_distractor(list_score,score_obj,obj_window,new_img,(x0,y0))

		hist_D=np.empty_like(hist_obj)
		hist_D[:]=0
		for num in range(len(dist_img_list)) :
			hist_dist=cv2.calcHist([dist_img_list[num]],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
			hist_D=hist_D+hist_dist

		prob_D=prob_obj(hist_obj,hist_D)
		prob_S=prob_obj(hist_obj,hist_bg)
		prob_comb_new=prob_S*lamda + prob_D*(1-lamda)
		color_map_new,prob_img_new=likelihood_map(prob_comb_new,new_img) # call likelihood function for obj-surr model
		color_map_final_updated,final_map_updated,prob_comb_new_updated=compute_maps(obj_window,new_img,hist_obj,prob_img_new,"rgb",prob_S,flag=1,anno="anno")
		prob_comb=update_model(prob_comb,prob_comb_new_updated)
		prob_comb6.append(prob_comb)
		counter=counter+1

	return top6_window,prob_comb6

def prob_obj(hist1,hist2) :
	'''This function creates a look-up table that contains the probability associated with the possible bin values.
	   In our case total bins=10*10*10. This thing will be computed for each frame. Then, when we need to localize the object
	   in the next frame, it will be used.
	   Input: histogram of object,surr/distractors
	   output:array of size 10x10x10 containing probability values for the corresponding bin. This array is called object model.'''
	global bin
	prob=np.zeros((bin,bin,bin),dtype='float32')
	print bin
	for i in range(bin) :
		for j in range(bin) :
			for k in range(bin) :
				if hist1[i][j][k]>0 or hist2[i][j][k] >0 :
					prob[i][j][k]= hist1[i][j][k]/(hist1[i][j][k]+hist2[i][j][k])
				else :
					prob[i][j][k]=0.5

	return prob

def min_func(alpha,i,obj_window) :
	global x_y_list,DP_cost,DP_score,DP_lastpos,sigma_square
	x2,y2,w,h=obj_window
	k=0
	i=int(i)
	cost_min=float('inf')
	while(k<6) :
		x1,y1,w1,h1=x_y_list[k][i-1]
		c_x=x1+(w/2)
		c_y=y1+(h/2)
		window2=np.mgrid[y2:y2+h,x2:x2+w]
		diff1=window2[0]-c_y
		diff2=window2[1]-c_x
		diff3=-(diff1*diff1 + diff2*diff2)
		distance_wt = np.exp(diff3/sigma_square)
		distance_wt = np.sum(distance_wt)
		#print DP_score[alpha][i]-distance_wt
		cost=-math.log(DP_score[alpha][i]+distance_wt) + DP_cost[k][i-1]
		if(cost<cost_min) :
			cost_min=cost
			box_no=k
		k=k+1	
	DP_cost[alpha][i]=cost_min
	DP_lastpos[alpha][i]=box_no+1

def get_distractor(list_score,score_obj,object_window,image,init_pt) :
	global lamda_v
	x0,y0=init_pt
	x,y,w,h=object_window
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

def compute_maps(object_window,image,hist_obj,prob_img_surr,image_type,prob_S,flag,anno) : # image- either lbp image or the original image(when rgb) 
	# flag=0 if search_window=3*object-window and flag=1 if search window is entire image
	global h_img,w_img,lamda,integral_images
	obj_cand=object_window
	x,y,w,h=object_window
	integral_image=cv2.integral(prob_img_surr,sdepth=-1)
	integral_image=(integral_image.astype('float'))/255
	if anno=="anno" :
		integral_images.append(integral_image)
	score_obj=vote_score(obj_cand,integral_image) # in this case obj_cand is the annotated object in the last frame itself
	print score_obj,"object score"
	#looking for distractors
	# setting variables for search window.. Also putting constraints like
	# variables shouldn't go beyond width/height of image
	# leftmost and topmost corner
	if flag==0 :
		start_x=int(max(0,x-w)) 
		start_y=int(max(0,y-h))
		# rightmost and bottommost corner
		end_x=int(min(x+w,w_img-w))
		end_y=int(min(y+h,h_img-h))
	else :
		start_x=0
		start_y=0
		end_x  =w_img-h
		end_y  =h_img-h	
	y0=start_y
	x0=start_x
	list_score=np.zeros((end_y-y0,end_x-x0),dtype='float') # array for containing vote-scores of all the candidates, will be used for updating distractors
	# score for all the candidates is computed and simultaneously filing the above array
	# Also, we calculate the obj-cand with max score 
	while (start_x<end_x) :
		if flag==0 :
			start_y=int(max(y-h,0))
		else :
			start_y=0	
		while(start_y<end_y) :
			obj_cand=(start_x,start_y,w,h)
			score=vote_score(obj_cand,integral_image)  # vote-score for the combined object candidate
			list_score[start_y-y0][start_x-x0]=score
			start_y=start_y+5 # sampling after 5 pixels
		start_x=start_x+5	

	# checking the condition for an object cand to be a distractor
	#####
	dist_img_list = get_distractor(list_score,score_obj,object_window,image,(x0,y0))
	hist_D=np.empty_like(hist_obj)
	if image_type=="rgb" :
		hist_D[:,:,:]=0
		for num in range(len(dist_img_list)) :
			hist_dist=cv2.calcHist([dist_img_list[num]],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
			hist_D=hist_D+hist_dist
		prob_D=prob_obj(hist_obj,hist_D)	
		color_map_dist,Prob_dist=likelihood_map(prob_D,image)# call likelihood function for obj-dist model
	
	Prob_comb=prob_img_surr.astype('float32')*(1-lamda)+Prob_dist.astype('float32')*lamda
	prob_comb=prob_D*lamda + prob_S*(1-lamda)
	final_map=Prob_comb.astype('uint8')
	color_map_final=cv2.applyColorMap(final_map, cv2.COLORMAP_JET)

	return color_map_final,final_map,prob_comb

if __name__ == "__main__":
	argument=sys.argv
	cap=cv2.VideoCapture("theater.mp4")
	var=0
	if (len(argument)<2) :
		print "\n \n provide an image as input\n\n"
		if var==1 :
			folder_name="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/sequence"
			image=cv2.imread(folder_name +"/00000001.jpg")
			newpath="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/result_trellis"
			if not os.path.exists(newpath) :
				os.makedirs(newpath)
		else :	
			ret,image=cap.read()
			frame_count=cap.get(7)
			num_images=frame_count-2
			image=cv2.resize(image,(400,250)) 

	if (len(argument)==2):	
		image=cv2.imread(str(argument[1])) # complete image of the scene

	if var==1 :
		path, dirs, files = os.walk(folder_name).next()
		num_images = len(files)
		filenames=["%08d" % number for number in range(num_images)]
		num_images = len(glob.glob1(folder_name,"*.jpg"))		
	prob_anno=[]
	object_window_list=[]
	color_maps=[]
	#print num_images,"num_images"
	integral_images=[]
	for i in range(2) :
		
		if i>0 :
			if var==1 :
				image=cv2.imread(folder_name+"/"+filenames[num_images-1]+".jpg")
			else :
				cap.set(cv2.CAP_PROP_POS_FRAMES,num_images-1)
				ret,image=cap.read()
				image=cv2.resize(image,(400,250))
			list_refpt=[]
		print "FRAME COUNT ",cap.get(cv2.CAP_PROP_POS_FRAMES)
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
		prob_S=prob_obj(hist_obj,hist_bg)
		color_map_surr,prob_img_surr=likelihood_map(prob_S,image) # call likelihood function for obj-surr model
		
		color_map_final,final_map,prob_comb=compute_maps(object_window,image,hist_obj,prob_img_surr,"rgb",prob_S,flag=0,anno="anno")

		### updating distractors
		color_map_final_updated,final_map_updated,prob_comb_updated=compute_maps(object_window,image,hist_obj,final_map,"rgb",prob_S,flag=1,anno=None)
		color_maps.append(color_map_final)
		cv2.imshow("obj-surr model",color_map_surr)
		#cv2.imshow("distractor-aware model",color_map_dist)
		cv2.imshow("combined",color_map_final)
		cv2.imshow("original_frame",frame)
		cv2.imshow("updated",color_map_final_updated)
		cv2.waitKey(0)	
		cv2.destroyAllWindows()

		prob_anno.append(prob_comb_updated)
	print len(prob_anno),"anno length"
	prob_comb=(prob_anno[0]+prob_anno[1])/2

	# here, I'll update distractors
	if var==0 :
		cap.set(cv2.CAP_PROP_POS_FRAMES,0)
		ret,image=cap.read()
		image=cv2.resize(image,(400,250)) 
	else :
		image=cv2.imread(folder_name +"/00000001.jpg")	
	color_map_final,Prob_comb=likelihood_map(prob_comb,image)
	color_map0,Prob_comb=likelihood_map(prob_anno[0],image)
	color_map1,Prob_comb=likelihood_map(prob_anno[1],image)
	cv2.imshow("combined colormap",color_map_final)
	cv2.imshow("colormap0",color_map0)
	#cv2.imshow("colormap00",color_maps[0])
	#cv2.imshow("colormap1",color_map1)
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
	obj_cand=(x,y,w,h)
	score=vote_score(obj_cand,integral_images[1]) # in this case obj_cand is the annotated object in the last frame itself
	DP_score[0][num_images-1]=score
	#print i,"it is i it is i"
	# for the very first frame, filling in DP array
	object_window=object_window_list[0]
	x,y,w,h=object_window
	obj_cand=(x,y,w,h)
	score=vote_score(obj_cand,integral_images[0]) # in this case obj_cand is the annotated object in the first frame itself
	DP_score[0][0]=score # other rows are of no use to us for the first frame as there is only one obj_cand here
	DP_lastpos[0][0]=1
	
	DP_cost[0][0]=-math.log(score) # normalized score has been used
	obj_prev_window=[object_window]
	#print object_window,"this is window",i
	prob_comb6_prev=[prob_comb]
	x_y_list=[[object_window],[object_window],[object_window],[object_window],[object_window],[object_window]]
	i=1
	if var==0 :
		cap.set(cv2.CAP_PROP_POS_FRAMES,1)

#--------------------------------------------------------------------------------------------------------------------------------------
   ##### the main while loop

	while(i<num_images-1) :
		t=time.time()
		# var variable is 1 when reading from a image files..if reading from a video set var=0 
		if var==1 :
			new_img=cv2.imread(folder_name+"/"+filenames[i]+".jpg")
		else :
			ret,new_img=cap.read()
			new_img=cv2.resize(new_img,(400,250))

		if(new_img==None) :
			break
		count=0
		current_windows6=[]
		current_prob6=[]
		scores6=[]		
		while(count<6 and DP_lastpos[count][i-1]!=0) :
			if(i==1) :
				number=6
			else :
				number=1	
			#print "count",count, prob_comb6_prev[count].shape,len(prob_comb6_prev)	
			list_obj_window,prob_comb6=obj_cand_new6(obj_prev_window[count],new_img,prob_comb6_prev[count],number)
			#print len(prob_comb6),prob_comb6[0].shape,"prob shapes"
			current_windows6.append(list_obj_window)
			current_prob6.append(prob_comb6)
			scores6.append([score[1] for score in list_obj_window])
			count=count+1
		#print len(list_obj_window),len(prob_comb6),len(current_windows6)
		#print current_windows6	
		scores6=np.array(scores6)
		scores6=scores6.ravel()
		alpha=0
		while(alpha<6) :
			DP_score[alpha][i]=scores6[alpha]
			if i==1 :
				DP_lastpos[alpha][i]=1
				x1,y1,w,h=obj_prev_window[0]
				x2,y2,w,h=current_windows6[0][alpha][0]
				c_x=x1+(w/2)
				c_y=y1+(h/2)
				window2=np.mgrid[y2:y2+h,x2:x2+w]
				diff1=window2[0]-c_y
				diff2=window2[1]-c_x
				diff3=-(diff1*diff1 + diff2*diff2)
				distance_wt = np.exp(diff3/sigma_square)
				distance_wt = np.sum(distance_wt)
				pos=DP_lastpos[alpha][i]-1
				DP_cost[alpha][i] = -math.log(DP_score[alpha][i] + distance_wt) + DP_cost[pos][i-1]
			else :
				# here we will need to minimise the functon over k
				min_func(alpha,i,current_windows6[alpha][0][0])
					
			alpha=alpha+1	
		obj_prev_window=[]
		prob_comb6_prev=[]
		alpha=0
		recta_image=copy.deepcopy(new_img)
		while(alpha<6) :
			if(i==1) :
				x_y_list[alpha].append(current_windows6[0][alpha][0])
				obj_prev_window.append(current_windows6[0][alpha][0])
				prob_comb6_prev.append(current_prob6[0][alpha])
				a,b,c,d = current_windows6[0][alpha][0]
				recta_image=cv2.rectangle(recta_image,(a,b),(a+c,b+d),(0,255,0),1)
				#print "watch this",x_y_list[alpha][i-1],current_windows6[0][alpha][0]
				cv2.imshow("new_img_____",recta_image)
				k=cv2.waitKey(10) & 0xff
			
			else :
				a,b,c,d = current_windows6[alpha][0][0]
				recta_image=cv2.rectangle(recta_image,(a,b),(a+c,b+d),(0,255,0),1)
				#print "watch this",x_y_list[alpha][i-1],current_windows6[alpha][0][0]
				cv2.imshow("new_img_____",recta_image)
				k=cv2.waitKey(10) & 0xff
				if alpha==0 :
					cv2.imwrite("result2/"+str(i)+".jpg",recta_image)
				x_y_list[alpha].append(current_windows6[alpha][0][0])	
				obj_prev_window.append(current_windows6[alpha][0][0])
				prob_comb6_prev.append(current_prob6[alpha][0])
			alpha=alpha+1	

		cv2.imshow("new_img",new_img)
		k=cv2.waitKey(10) & 0xff
		if k==27 :
			break
		i=i+1
	cv2.destroyAllWindows()
	# for final frame
	object_window=object_window_list[1]
	print object_window
	min_func(0,num_images-1,object_window)
	print DP_cost.shape,"cost shape"
	min_path=[object_window]
	count=int(num_images)-1
	row=0
	while(count>0) :
		row=DP_lastpos[row][count]-1
		count=count-1
		obj_win=x_y_list[row][count]
		min_path.append(obj_win)
	min_path1=min_path.reverse()
	min_path1= min_path[::1]
	cap.set(cv2.CAP_PROP_POS_FRAMES,0)
	for i in range(0,int(num_images)) :
		if var==0 :
			ret,new_img=cap.read()
			new_img=cv2.resize(new_img,(400,250))
		else :
			new_img=cv2.imread(folder_name+"/"+filenames[i+1]+".jpg")	
		if new_img==None :
			break
			
		x,y,w,h=x_y_list[0][i]
		cv2.rectangle(new_img,(x,y),(x+w,y+h),(0, 255, 0), 2) 
		cv2.imwrite("result1/" + str(i)+".jpg",new_img)
		cv2.imshow("new",new_img)
		cv2.waitKey(10)
		
	#print np.argmin(DP_cost[:,num_images-1])
	cv2.destroyAllWindows()  	  