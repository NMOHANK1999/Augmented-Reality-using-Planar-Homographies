# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:04:21 2023

@author: NIshanth Mohankumar
"""

import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

#try placing a gaussian filter here to see if it improves the model.


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    A = []
    for i in range(len(x1)):
        A.append([-x1[i,0], -x1[i,1], -1, 0, 0, 0, x2[i,0]*x1[i,0], x2[i,0]*x1[i,1], x2[i,0]])  #should the last 3 values be negative instead? 
        A.append([0, 0, 0, -x1[i,0], -x1[i,1], -1, x2[i,1]*x1[i,0], x2[i,1]*x1[i,1], x2[i,1]])
        
    A = np.array(A)
    a, b, At = np.linalg.svd(A)
    #print(a, b, At)
    H2to1 = At[-1, :].reshape(3, 3)
    return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
    #assuming that x1 and x2 are N x 2
	#Compute the centroid of the points
    centroid1 = np.mean(x1, axis = 0)  #may have to put it as axis= 0 if we convert to row mat
    centroid2 = np.mean(x2, axis = 0)  

	#Shift the origin of the points to the centroid
    translated_x1 = x1 - centroid1
    translated_x2 = x2 - centroid2
    
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    scale_x1 = 2**(0.5) / np.max(translated_x1, axis = 1) #is this axis = 0?????????
    scale_x2 = 2**(0.5) / np.max(translated_x2, axis = 1)
    
	#Similarity transform 1
    #below I have made the transform for the translation for points 1 and 2
    T1 = [[scale_x1, 0, -scale_x1 * centroid1[0]],
          [0, scale_x1, -scale_x1 * centroid1[1]],
          [0, 0, 1]]

	#Similarity transform 2
    T2 = [[scale_x2, 0, -scale_x2 * centroid2[0]],
          [0, scale_x2, -scale_x2 * centroid2[1]],
          [0, 0, 1]]


    T1 = np.array(T1)
    T2 = np.array(T2)
    
    #new values of x1, and x2
    X1 =  np.row_stack([x1, np.ones(x1.shape[0])])
    X2 =  np.row_stack([x2, np.ones(x2.shape[0])])
    
    norm_x1 = T1 @ X1
    norm_x2 = T2 @ X2
                         
    norm_x1 = (norm_x1.T)[:, 0:2]
    norm_x2 = (norm_x2.T)[:, 0:2]
    
	#Compute homography
    H_norm = computeH( norm_x1 , norm_x2 )


	#Denormalization, what happens if T1 is singular??
    H2to1 =  np.linalg.inv(T1) @ H_norm @ T2
   
    return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    
    #here the assumption is that the points are already matched(len(locs1) == len(locs2))
    best_inlier = 0
    bestH2to1 = None
    #looping for iterations
    for i in range(max_iters):
        random_ind = np.random.choice(len(locs1), 4)
        random_x1 = locs1[random_ind] #4x2
        random_x2 = locs2[random_ind] #4x2
        
        H2_1 = computeH_norm(random_x1, random_x2) #3X3
        
        # locs is a Nx2 numpy matrix so lets convert it to a 3xN homogenous mat for conversion
        
        random_homogerous_x1 = np.row_stack([random_x1.T, np.ones(len(random_x1))]) #3x4
        random_homogerous_x2 = np.row_stack([random_x2.T, np.ones(len(random_x2))]) #3x4
        
        #computing the transformation
        computed_homogeous_x1 = H2_1 @ random_homogerous_x2 #3x4
        computed_inhomogeous_x1 = computed_homogeous_x1[:2, :] / computed_homogeous_x1[2, :] #converting to inhomogenous 2x4
        
        
        
        #checking the accuracy with inline
        errors = np.linalg.norm(computed_inhomogeous_x1 - random_x1, axis=0)  
        inliers = np.sum(errors < inlier_tol)

        print(errors < inlier_tol)
        
        
        #replacing best model
        if inliers > best_inlier:
            best_inlier = inliers
            bestH2to1 = H2_1


    #changed inliers to best_inlier
    return bestH2to1, best_inlier



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography
    
    height = template.shape[0]
    width =  template.shape[1]

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
    

	#Create mask of same size as template
    mask = np.ones_like(template)    

	#Warp mask by appropriate homography
    warped_mask_image = cv2.warpPerspective(mask, H2to1, (width, height))

	#Warp template by appropriate homography
    warped_template_image = cv2.warpPerspective(img, H2to1, (width, height)) #sould this be image or template

	#Use mask to combine the warped template and the image
    composite_img = template.copy()
    composite_img[warped_mask_image == 1] = warped_template_image[warped_mask_image == 1]
	
    return composite_img




def matchPics(I1, I2, ratio, sigma):
    #I1, I2 : Images to match
    #opts: input opts
    #ratio for BRIEF feature descriptor
    
    
	
	#Convert Images to GrayScale
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    
	#Detect Features in Both Images
    locs1 = corner_detection(I1_gray, sigma)
    locs2 = corner_detection(I2_gray, sigma)
    
	#Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1_gray, locs1)
    desc2, locs2 = computeBrief(I2_gray, locs2)
    
	#Match features using the descriptors
    matches = briefMatch(desc1,desc2,ratio)
    
     #this is not needed, go to office hours and ask: where this is needed?
    locs1[:, [0, 1]] =  locs1[:, [1, 0]]     
    locs2[:, [0, 1]] =  locs2[:, [1, 0]]
    
    return matches, locs1, locs2
 


#Write script for Q2.2.4
opts = get_opts()

#1. Reads cv cover.jpg, cv desk.png, and hp cover.jpg.
cv_cover = cv2.imread('../data/cv_cover.jpg')  
cv_desk = cv2.imread('../data/cv_desk.png')  
hp_cover = cv2.imread('../data/hp_desk.png') 

#2. Computes a homography automatically using MatchPics and computeH ransac.

#check if i have to switch location of hp_cover and cv_desk or switch the locs1_matched with locs1_matched
# in the computeH_ransac func 
matches, locs1, locs2 = matchPics(cv_cover, hp_cover, opts)  

 
locs1_matched = locs1[matches[:, 0]]
locs2_matched = locs2[matches[:, 1]]


H2_1, best_inlier = computeH_ransac(locs1_matched, locs2_matched, opts)

#3. Uses the computed homography to warp hp cover.jpg to the dimensions of the
#   cv desk.png image using the OpenCV function cv2.warpPerspective function.

# the arguments are the source image, then the H then its the shape of the destination image(output image), is it 0,1 or 1,0

#4. At this point you should notice that although the image is being warped to the
#correct location, it is not filling up the same space as the book. Why do you think
#this is happening? How would you modify hp cover.jpg to fix this issue?
composite_image = compositeH(H2_1, hp_cover, cv_cover)   #cv_cover = template and cv_desk = img, we're warping img to template


cv2.imshow('Warped Img on template ', composite_image)
cv2.imshow('Warped Img on template ', hp_cover)
cv2.waitKey(0)
cv2.destroyAllWindows()


#5. Implement the function:
#composite img = compositeH( H2to1, template, img )
#to now compose this warped image with the desk image as in in Figure 4



#6. Include your result in your write-up



