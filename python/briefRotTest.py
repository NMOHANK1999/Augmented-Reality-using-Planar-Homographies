import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from scipy import ndimage
from matplotlib import pyplot as plt
from helper import plotMatches

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')


# cv2.imshow('cv_cover', cv_cover)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# rot_cv_cover =  ndimage.rotate(cv_cover, 50)  
# cv2.imshow('rotated_cv_cover', rot_cv_cover)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

matches_for_angle = [] 
for i in range(36):
	#Rotate Image
    rot_cv_cover =  ndimage.rotate(cv_cover, i * 10)   
	#Compute features, descriptors and Match feature
    matches,locs1,locs2 = matchPics(cv_cover, rot_cv_cover, opts)
	#Update histogram
    matches_for_angle.append(len(matches))

    print(i*10, " degrees")
    plotMatches(cv_cover, rot_cv_cover, matches, locs1, locs2)


#Display histogram
angles = np.arange(0, 360, 10)
plt.bar(angles, matches_for_angle, width=8, align='edge', alpha=0.7)
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.title('Histogram of Matches for Different Orientations')
plt.show()
