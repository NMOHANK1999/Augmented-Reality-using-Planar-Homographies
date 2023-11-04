import numpy as np
import cv2
#Import necessary functions

import skimage.io 
import skimage.color
from opts import get_opts
from multiprocessing import Pool
from functools import partial

#Import necessary functions

from matchPics import matchPics
from helper import plotMatches
from planarH import computeH
from planarH import computeH_norm
from planarH import computeH_ransac
from planarH import compositeH
from loadVid import loadVid

opts = get_opts()
#Write script for Q3.1

#the template is sent to the image, change btw the args and arg_tuple

def process_frame(cv_cover, template_video, image_video, opts):
    #template_video, image_video, opts = args_tuple
    matches, locs1, locs2 = matchPics(image_video, cv_cover, opts)

    locs1_matched = locs1[matches[:, 0]]
    locs2_matched = locs2[matches[:, 1]]
    
    H2_1, best_inlier = computeH_ransac(locs1_matched, locs2_matched, opts)
    print(H2_1)
    
    cover_height, cover_width, _ = cv_cover.shape

    height_temp = template_video.shape[0]

    width_temp = template_video.shape[1]

    conv_cov_width = cover_width * (height_temp / cover_height)

    template = template_video[70:-70, int((width_temp - conv_cov_width) // 2) : int((width_temp + conv_cov_width) // 2), : ]

    template = cv2.resize(template, [cover_width, cover_height])
    
    composite_image = compositeH(H2_1, template, image_video)
    
    return composite_image



#get the frams for both videos
#declare the book= image (the destination)
book_img = loadVid('../data/book.mov')

#declare the panda template (the source)
arsource_temp = loadVid('../data/ar_source.mov') 


''' ATTEMPTED VIA MULTIPROCESSISNG
frames_img = book_img.shape[0]
frames_temp, height_temp, width_temp, _ = arsource_temp.shape
frames = min(frames_img, frames_temp) 

#comment out after check
frames = 20 
# 

book_img = book_img[:frames, :, :, :]
arsource_temp = arsource_temp[:frames, :, :, :]


#here i will crop the middle section of the video out using the cv_cover dimensions 
cv_cover = cv2.imread('../data/cv_cover.jpg')


frame_args_list = [(cv_cover, arsource_temp[i, :, :, :], book_img[i, :, :, :], opts) for i in range(frames)]

with Pool() as pool:
        output_frames = pool.map(process_frame, frame_args_list)

    
out_path = '../data/ar.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, 20.0, (book_img.shape[2], book_img.shape[1]))

# Write the frames to the output video
for frame in output_frames:
    out.write(frame)

# Release the video writer and display a message
out.release()
print(f'AR video saved at {out_path}')



'''
"""#FOR WRITE UP
# delete this after testing
temp = arsource_temp[200, :, :, :]
img = book_img[200, :, :, :]

cv_cover = cv2.imread('../data/cv_cover.jpg')


comp = process_frame(cv_cover, temp, img, opts)

# cv2.imshow('cv_cover', temp)
# cv2.imshow('comp Img on template ', comp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite("../data/comp1.jpg", comp)

"""

#WITHOUT MULTIPROCESSING

cv_cover = cv2.imread('../data/cv_cover.jpg')

frames_img = book_img.shape[0]
frames_temp, height_temp, width_temp, _ = arsource_temp.shape
frames = min(frames_img, frames_temp) 


book_img = book_img[:frames, :, :, :]
arsource_temp = arsource_temp[:frames, :, :, :]

resultant = []
for i in range(frames):
    comp = process_frame(cv_cover, arsource_temp[i, :, :, :], book_img[i, :, :, :], opts)
    resultant.append(comp)
    
resultant = np.array(resultant)
video = cv2.VideoWriter("../data/ar.avi", 0, 30, (resultant.shape[2], resultant.shape[1]))
for frame in resultant:
    video.write(frame)

cv2.destroyAllWindows()
video.release()

