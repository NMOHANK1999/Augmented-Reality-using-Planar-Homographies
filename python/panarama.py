# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 01:41:58 2023

@author: NIshanth Mohankumar
"""
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
from opts import get_opts

opts = get_opts()

left = cv2.imread('../data/pano_left.jpg')
right = cv2.imread('../data/pano_right.jpg')

left = cv2.resize(left, (left.shape[1], left.shape[0]))
right = cv2.resize(right, (right.shape[1], right.shape[0]))

'''
This is via the SIFT method for feature point etraction
'''

sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(left,None)
kp2, des2 = sift.detectAndCompute(right,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w, _ = left.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(right,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found ")
    matchesMask = None
    
#4. Running RANSAC
#H2_1, best_inlier = computeH_ransac(locs1_matched, locs2_matched, opts)

panaroma = compositeH(M, left, right)

'''
This is via the FAST+BRIEF method for feature point etraction
'''
matches, locs1, locs2 = matchPics(left, right, opts)  

#3. finding the locatios corresponding
locs1_matched = locs1[matches[:, 0]]
locs2_matched = locs2[matches[:, 1]]

H2_1, best_inlier = computeH_ransac(locs1_matched, locs2_matched, opts)

panaroma2 = compositeH(H2_1, right, left)



cv2.imshow('Using SIFT ', panaroma)
cv2.imshow('Using BRIEF and FAST ', panaroma2)
cv2.waitKey(0)
cv2.destroyAllWindows()
