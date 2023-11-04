import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection



def matchPics(I1, I2, opts): #Nishanth
    #I1, I2 : Images to match
    #opts: input opts
    #ratio for BRIEF feature descriptor
    
    
    ratio = opts.ratio
    #threshold for corner detection using FAST feature detector
    sigma = opts.sigma  
	
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
