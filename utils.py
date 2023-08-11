import cv2 
import numpy as np
import copy
#from srhen import HNet
import torch

try :
    import kornia as K
    import kornia.feature as KF
    from kornia_moons.feature import *
    from kornia_moons.viz import draw_LAF_matches
except:
    print("you cannot import kornia so you can not use LOFTR!")


class FeatureExtraction:
    def __init__(self, img,method_name):
        self.img = copy.copy(img)
        self.gray_img = img


        if method_name == "ORB":
            self.method = cv2.ORB_create(
                nfeatures=400,
                scaleFactor=1.2,
                scoreType=cv2.ORB_HARRIS_SCORE)
        if method_name == "SIFT":
            self.method = cv2.xfeatures2d.SIFT_create(400)
        if method_name == "SURF":
           self.method = cv2.xfeatures2d.SURF_create(2200)

        if len(img.shape) == 3:
            self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.kps, self.des = self.method.detectAndCompute( \
            self.gray_img, None)
        self.img_kps = cv2.drawKeypoints( \
            self.img, self.kps, 0, \
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.matched_pts = []



def get_features(img0,img1,method):
    features0 = FeatureExtraction(img0,method)
    features1 = FeatureExtraction(img1,method)

    matches = feature_matching(features0, features1,method)
    assert matches != None ,"no matches found"


    # print("[INFO] template image # of feature points : {}".format(len(features0.kps)))
    # print("[INFO] destination image # of feature points : {}".format(len(features1.kps)))

    matched_image = cv2.drawMatches(img0, features0.kps, 
        img1, features1.kps, matches, None, flags=2)

    # print("[INFO] # of matched points : {} {}".format(len(features1.matched_pts),len(features0.matched_pts)))
    assert len(features0.matched_pts) >= 4, "matched points must be at least 4"
    H, _ = cv2.findHomography( features0.matched_pts, 
        features1.matched_pts, cv2.RANSAC, 5.0)
    # print("Homography Matrix : \n",H)

    # cv2.imshow("template image", img0)
    # cv2.imshow("destination image", img1)
    # cv2.imshow("matched image", matched_image)
    # cv2.waitKey(0)
    return H,features0,features1,matched_image




def feature_matching(features0, features1,method_name):
    if method_name in ["SIFT","SURF"]:
        index_params= dict(algorithm =5 , #FLANN_INDEX_KDTREE,
                        table_number = 6, # 12
                        key_size = 10,     # 20
                        multi_probe_level = 2) #2
    else:
        index_params= dict(algorithm = 6, #FLANN_INDEX_LSH
                        table_number = 6, # 12
                        key_size = 10,     # 20
                        multi_probe_level = 2) #2
    # index_params = dict(
    #     algorithm = 6, # FLANN_INDEX_LSH
    #     table_number = 6,
    #     key_size = 10,
    #     multi_probe_level = 2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params)
    LOWES_RATIO = 0.9
    MIN_MATCHES = 10
    matches = [] # good matches as per Lowe's ratio test
    if(features0.des is not None and len(features0.des) > 2):
        all_matches = flann.knnMatch( features0.des, features1.des, k=2)
        try:
            for m,n in all_matches:
                if m.distance < LOWES_RATIO * n.distance:
                    matches.append(m)
        except ValueError:
            pass
        if(len(matches) > MIN_MATCHES):    
            features0.matched_pts = np.float32( 
                [ features0.kps[m.queryIdx].pt for m in matches ]  
                ).reshape(-1,1,2)
            features1.matched_pts = np.float32( 
                [ features1.kps[m.trainIdx].pt for m in matches ] 
                ).reshape(-1,1,2)
    return matches



def optical_flow_method(prev_gray, curr_gray):
    # Detect feature points in the previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)

    # Calculate optical flow (i.e., track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

   # Calculate homography matrix
    H, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC)

    return prev_pts, curr_pts,H

def srhenet_method(prev_gray,curr_gray,model):
        patch_size = (128, 128)
        down_ratio = 1
        resized_w = prev_gray.shape[1] // down_ratio
        resized_h =curr_gray.shape[0] //down_ratio

        #resize 
        prev_gray_resized = cv2.resize(prev_gray,(resized_w,resized_h))
        curr_gray_resized = cv2.resize(curr_gray,(resized_w,resized_h))

        center_x = prev_gray_resized.shape[1] // 2
        center_y = curr_gray_resized.shape[0] // 2

        # Define the coordinates of the top-left corner of the patch
        x = center_x - patch_size[0] // 2
        y = center_y - patch_size[1] // 2

        # Extract the patch from the previous grayscale image using NumPy indexing
        prev_patch = prev_gray_resized[y:y+patch_size[1], x:x+patch_size[0]]

        # Extract the patch from the current grayscale image using NumPy indexing
        curr_patch = curr_gray_resized[y:y+patch_size[1], x:x+patch_size[0]]

        # Normalize
        prev_patch = prev_patch.astype(np.float32) / 255.0
        curr_patch = curr_patch.astype(np.float32) / 255.0
    
        # ToTensor
        prev_patch = np.expand_dims(prev_patch, axis=0)
        prev_patch = torch.from_numpy(prev_patch).unsqueeze(0).to(torch.float32)
    
        curr_patch = np.expand_dims(curr_patch, axis=0)
        curr_patch = torch.from_numpy(curr_patch).unsqueeze(0).to(torch.float32)

        dist = model(prev_patch,curr_patch).detach().cpu().numpy().reshape(4, 2)


        patch_size = patch_size[0]
        # Define the coordinates of the four corners of the patch
        top_left = (center_x - patch_size//2, center_y - patch_size//2)
        top_right = (center_x + patch_size//2, center_y - patch_size//2)
        bottom_left = (center_x - patch_size//2, center_y + patch_size//2)
        bottom_right = (center_x + patch_size//2, center_y + patch_size//2)

        # Store the coordinates of the four corners in a NumPy array
        corners = np.array([top_left, top_right, bottom_left, bottom_right],dtype=np.float32)
        dist_corners = corners + dist
        


        # #Draw the rectangle on the copy of the image
        # cv2.polylines(prev_gray, np.int32([corners]), True, (0, 255, 0), 2)
        # #Draw the rectangle on the image
        # cv2.rectangle(prev_gray, top_left, bottom_right, (0, 255, 0), 2)
        # cv2.imshow("Image with rectangle", prev_gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        #H_srhen, _ = cv2.findHomography(corners, dist_corners)

        # H_gt,features0,features1,matched_image =get_features(prev_gray_resized,curr_gray_resized,"ORB")
        # dist_corners_gt = cv2.perspectiveTransform(corners.reshape(4,1,2), H_gt) 

        # # Compute the Euclidean distance between each pair of corners
        # ace = np.linalg.norm(dist_corners.reshape(4,2) - dist_corners_gt.reshape(4,2), axis=1)
        # # Compute the Mean Absolute Corner Error
        # ace = np.mean(ace)
        # sum_ace = sum_ace +ace
        # print("ace : ",ace)

        

        # prev_pts = features0.matched_pts.reshape(-1,1,2)
        # curr_pts = cv2.perspectiveTransform(prev_pts, H_gt) 

        prev_pts = corners.reshape(4,1,2)
        curr_pts = dist_corners.reshape(4,1,2) #dist_corners_gt.reshape(4,1,2)

        # Calculate homography matrix
        H, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC)

        return prev_pts,curr_pts,H

def loftr_method(prev_gray,curr_gray,model):
        #resize
        down_ratio = 4
        resized_w = prev_gray.shape[1] // down_ratio
        resized_h =curr_gray.shape[0] //down_ratio

        #resize 
        prev_gray_resized = cv2.resize(prev_gray,(resized_w,resized_h))
        curr_gray_resized = cv2.resize(curr_gray,(resized_w,resized_h))

        loftr_input1 = prev_gray_resized
        loftr_input2 = curr_gray_resized
        loftr_input1 = torch.tensor(loftr_input1).unsqueeze(0).unsqueeze(0) /255.0
        loftr_input2 = torch.tensor(loftr_input2).unsqueeze(0).unsqueeze(0) /255.0
        
        
        input_dict = {
            "image0": loftr_input1, #K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
            "image1": loftr_input2 #K.color.rgb_to_grayscale(img2),
        }


        with torch.inference_mode():
            correspondences = model(input_dict)


        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0



        
        mkpts0_inliers = [[ round(num) for num in x] for x, m in zip(mkpts0, inliers) if m]
        mkpts1_inliers = [[ round(num) for num in x] for x, m in zip(mkpts1, inliers) if m]


        # Estimate homography using RANSAC
        H, _ = cv2.findHomography(np.array(mkpts0_inliers), np.array(mkpts1_inliers), cv2.RANSAC)
        # print(Fm)
        # print(H)

        prev_pts = np.array(mkpts0_inliers,dtype=np.float32).reshape(-1,1,2)
        curr_pts = np.array(mkpts1_inliers,dtype=np.float32).reshape(-1,1,2)

        return prev_pts,curr_pts,H
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth_movingaverage(trajectory,smoothing_radius):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=smoothing_radius)

    return smoothed_trajectory

class KalmanFilter:
    def __init__(self, state_dim=3, measurement_dim=3):
        # Initialize the Kalman filter matrices
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.A = np.eye(self.state_dim)  
    
        # Measurement matrix      
        self.H = np.eye(self.measurement_dim)  
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.01

        # Measurement noise covariance    
        self.R = np.eye(self.measurement_dim) * 10

        # Initial state covariance
        self.P = np.eye(self.state_dim) * 1000

        # Initial state    
        self.x = np.zeros((self.state_dim, 1))

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Update the state estimate based on the measurement z
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.state_dim) - np.dot(K, self.H), self.P)

def smooth(trajectory, smoothing_radius):
    state_dim = 3
    measurement_dim = 3
    kf = KalmanFilter(state_dim, measurement_dim)
    smoothed_trajectory = np.zeros_like(trajectory)

    for i in range(len(trajectory)):
        # Predict the next state
        kf.predict()

        # Update the state estimate based on the measurement
        z = trajectory[i]
        kf.update(z)

        # Get the smoothed state estimate
        x = kf.x.squeeze()
        print(x[0])
        smoothed_trajectory[i] = x[0]

    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame




def translate_img(img,t_x=0,t_y=0):
  
    T = np.float32([[1, 0, t_x], [0, 1, t_y]])
    
    # We use warpAffine to transform
    # the image using the matrix, T
    img = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]))
    return img



def rotate_image(image, angle): #angle is in degrees
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotateAndScale(img, scaleFactor = 0.5, degreesCCW = 30):
    (oldX,oldY) = img.shape[:2] #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg


# Rotation
# Given two points p1 and p2

# p1 = H ⋅ p1

# p2 = H ⋅ p2

# Now just calculate the angle between vectors p1 p2 and p1' p2'
def calculate_rotation(index_pt1,index_pt2,matched_pts0,matched_pts1,H):
    pt1_in0 = matched_pts0[index_pt1]
    pt2_in0 = matched_pts0[index_pt2]

    #these are originals, we estimate in below using our H matrix
    pt1_in1 = matched_pts1[index_pt1]
    pt2_in1 = matched_pts1[index_pt2]

    # Apply homography matrix to point
    homogeneous_point1_in0 = np.array([pt1_in0[0], pt1_in0[1], 1])
    transformed_point1_in0 = H.dot(homogeneous_point1_in0)
    # Convert back to non-homogeneous coordinates
    ept1_in1 = (round(transformed_point1_in0[0] / transformed_point1_in0[2]), round(transformed_point1_in0[1] / transformed_point1_in0[2]))

    #do the same for 2nd
    homogeneous_point2_in0 = np.array([pt2_in0[0], pt2_in0[1], 1])
    transformed_point2_in0 = H.dot(homogeneous_point2_in0)
    ept2_in1 = (round(transformed_point2_in0[0] / transformed_point2_in0[2]), round(transformed_point2_in0[1] / transformed_point2_in0[2])) 

    ept1_in1,ept2_in1 = np.array(ept1_in1),np.array(ept2_in1)

    # Calculate vectors
    vec1 = pt2_in0 - pt1_in0
    vec2 = ept2_in1 - ept1_in1

    # Check for division by zero
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return np.nan

    # Calculate angle between vectors
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return angle


# Translation
# Given a single point, for translation do

# T = dst - (H ⋅ src)
def calculate_translation(index_pt1,index_pt2,matched_pts0,matched_pts1,H):
    pt1_in0 = matched_pts0[index_pt1]

    #these are originals, we estimate in below using our H matrix
    pt1_in1 = matched_pts1[index_pt1]

    # Apply homography matrix to point
    homogeneous_point1_in0 = np.array([pt1_in0[0], pt1_in0[1], 1])
    transformed_point1_in0 = H.dot(homogeneous_point1_in0)
    # Convert back to non-homogeneous coordinates
    ept1_in1 = (round(transformed_point1_in0[0] / transformed_point1_in0[2]), round(transformed_point1_in0[1] / transformed_point1_in0[2]))

    translation = ept1_in1 - pt1_in0

    return translation

# Scale
# Given two pairs of points p1, p2 and p1', p2'
# Compare the lengths: |p1 p2| and |p1' p2'|.
# To be fair, use another segment orthogonal to the first and average the result.
# You will see that there is no constant scale factor or translation one.
# They will depend on the src location.
def calculate_scale(index_pt1,index_pt2,matched_pts0,matched_pts1,H):
    pt1_in0 = matched_pts0[index_pt1]
    pt2_in0 = matched_pts0[index_pt2]

    #these are originals, we estimate in below using our H matrix
    pt1_in1 = matched_pts1[index_pt1]
    pt2_in1 = matched_pts1[index_pt2]

    # Apply homography matrix to points
    homogeneous_point1_in0 = np.array([pt1_in0[0], pt1_in0[1], 1])
    transformed_point1_in0 = H.dot(homogeneous_point1_in0)
    # Convert back to non-homogeneous coordinates
    ept1_in1 = (round(transformed_point1_in0[0] / transformed_point1_in0[2]), round(transformed_point1_in0[1] / transformed_point1_in0[2]))

    homogeneous_point2_in0 = np.array([pt2_in0[0], pt2_in0[1], 1])
    transformed_point2_in0 = H.dot(homogeneous_point2_in0)
    # Convert back to non-homogeneous coordinates
    ept2_in1 = (round(transformed_point2_in0[0] / transformed_point2_in0[2]), round(transformed_point2_in0[1] / transformed_point2_in0[2]))

    # Calculate the length of the two vectors
    length1 = np.linalg.norm(np.array(pt1_in0) - np.array(pt2_in0))
    length2 = np.linalg.norm(np.array(ept1_in1) - np.array(ept2_in1))

    # Check for division by zero
    if length1 == 0 or length2 == 0:
        return np.nan

    # Calculate the scale factor
    scale = length2 / length1

    return scale

def infer_srhen_model(model_file,device="cpu"):
    # Network
    hnet = HNet()
    hnet.eval()
    hnet.to(device)
    hnet.load_state_dict(torch.load(model_file, map_location = torch.device('cpu')))
    return hnet

def infer_loftr_model(pretrained_type="outdoor",device="cpu"):


    matcher = KF.LoFTR(pretrained=pretrained_type)
    matcher.eval()
    matcher.to(device)


    return matcher