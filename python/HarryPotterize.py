
import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt

#Import necessary functions

from matchPics import matchPics
from helper import plotMatches
from planarH import computeH
from planarH import computeH_norm
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q2.2.4
opts = get_opts()

#1. Reads cv cover.jpg, cv desk.png, and hp cover.jpg.
cv_cover = cv2.imread('../data/cv_cover.jpg')  #book
cv_desk = cv2.imread('../data/cv_desk.png')  #img
hp_cover = cv2.imread('../data/hp_cover.jpg') #template



#2. Computes matches btw the images.
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)  

#3. finding the locatios corresponding
locs1_matched = locs1[matches[:, 0]]
locs2_matched = locs2[matches[:, 1]]

#4. Running RANSAC
H2_1, best_inlier = computeH_ransac(locs1_matched, locs2_matched, opts)


print(H2_1)
height_cv_cover  = cv_cover.shape[0]
width_cv_cover  = cv_cover.shape[1]

#5 reshaping HP cover to fill full frame of the the book
hp_cover = cv2.resize(hp_cover, [width_cv_cover, height_cv_cover]) 

#6. creating the composite image
composite_image = compositeH(H2_1, hp_cover, cv_desk)   


#7. plotting the composite image
#cv2.imshow('Warped Img on template ', composite_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#writing image
cv2.imwrite("../data/composite_image.jpg", composite_image)