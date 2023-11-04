


import numpy as np
import cv2

#axis=0 along columns

def computeH(x1, x2): #Nishanth
	#Q2.2.1
	#Compute the homography between two sets of points
    A = []
    
    
    for i in range(len(x1)):
        A.append([-x2[i,0], -x2[i,1], -1, 0, 0, 0, x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0]])  #should the last 3 values be negative instead? 
        A.append([0, 0, 0, -x2[i,0], -x2[i,1], -1, x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1]])  #switch the x1 and x2
        
    A = np.array(A)
    
    
    _, _, At = np.linalg.svd(A)
    H2to1 = At[-1, :].reshape(3, 3)
    

    return H2to1



def computeH_norm(x1, x2): #nishanth
	#Q2.2.2
    #assuming that x1 and x2 are N x 2
	#Compute the centroid of the points
    
    centroid1 = np.mean(x1, axis = 0)  
    centroid2 = np.mean(x2, axis = 0)  

	#Shift the origin of the points to the centroid
    translated_x1 = x1 - centroid1
    translated_x2 = x2 - centroid2
    
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    scale_x1 = 2**(0.5) / np.max(np.linalg.norm(translated_x1, axis = 1)) 
    scale_x2 = 2**(0.5) / np.max(np.linalg.norm(translated_x2, axis = 1))
    
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
    #X1 =  np.column_stack([x1, np.ones(x1.shape[0])]).T
    #X2 =  np.column_stack([x2, np.ones(x2.shape[0])]).T
    
    #norm_x1 = T1 @ X1
    #norm_x2 = T2 @ X2
    norm_x1 = np.dot(T1, np.column_stack((x1, np.ones(x1.shape[0]))).T).T
    norm_x2 = np.dot(T2, np.column_stack((x2, np.ones(x2.shape[0]))).T).T
    
    #Inhomogenenous 
    norm_x1 = norm_x1[:, 0:2]
    norm_x2 = norm_x2[:, 0:2]

    #print("norm_x1 inhomo", norm_x1)
    
	#Compute homography
    H_norm = computeH( norm_x1 , norm_x2 )
    
    #print("h norm", H_norm)


	#Denormalization,
    H2to1 =  np.linalg.inv(T1) @ H_norm @ T2
    H2to1 = H2to1 / H2to1[-1,-1]
   
    return H2to1


def computeH_ransac(locs1, locs2, opts): 
    max_iters = opts.max_iters
    inlier_tol = opts.inlier_tol
    
    assert locs1.shape[0] == locs2.shape[0]
    
    bestH2to1 = None
    best_inliers = 0
    
    for i in range(max_iters):
        # Randomly picks up 4 points for x1 and x2
        indices = np.random.choice(locs1.shape[0], 4, replace=False)
        x1 = locs1[indices]
        x2 = locs2[indices]
        
        # Compute homography
        H2to1 = computeH_norm(x1, x2)
        
        # Calculates the inhomogenous coordnates of x1
        x2_hom = np.concatenate([locs2, np.ones((locs1.shape[0], 1))], axis=1)
        estimated_points_x1 = (H2to1 @ x2_hom.T).T
        estimated_points_x1 = estimated_points_x1[:, :2] / estimated_points_x1[:, 2, np.newaxis]
        
        error = np.linalg.norm(estimated_points_x1 - locs1, axis=1)
        inliers = np.sum(error < inlier_tol)
        
        # Update the best homography if the current one is better.
        if inliers > best_inliers:
            print(inliers)
            best_inliers = inliers
            bestH2to1 = H2to1
    
    return bestH2to1, best_inliers



def compositeH(H2to1, template, img): 
	
	#Create a composite image after warping the template image on top
	#of the image using the homography
    
    height = img.shape[0]
    width =  img.shape[1]

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
    

	#Create mask of same size as template
    mask = np.ones_like(template)    

	#Warp mask by appropriate homography
    warped_mask_image = cv2.warpPerspective(mask, H2to1, (width, height))[:,:,0]

	#Warp template by appropriate homography
    warped_template_image = cv2.warpPerspective(template, H2to1, (width, height)) #shouldnt this be warping the image?
    
	#Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[warped_mask_image == 1] = warped_template_image[warped_mask_image == 1]
	
    return composite_img

