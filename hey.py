#https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/


# Import numpy and OpenCV
import numpy as np
import cv2
from utils import *
import time

DEVICE = "cpu"

VIDEO_FILE = "input/drone2.mp4"

METHODS =["OPTICAL_FLOW","SIFT","ORB","SURF"] #,"LOFTR","SRHENET"]
smoothing_method = "mavg" #kalman, mavg


#assert METHOD in METHODS,"METHOD DOES NOT EXIST!"
assert smoothing_method in ["kalman","mavg"]


for METHOD in METHODS:
    if METHOD == "SRHENET":
        model  = infer_srhen_model("C:/Users/PC_4232/Desktop/can/SRHEN-main/model_weights/srhen2/model_45.pt",device=DEVICE)
    elif METHOD == "LOFTR":
        model = infer_loftr_model(pretrained_type="outdoor",device=DEVICE)

    # Read input video
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



    # Destroy the window
    cv2.destroyWindow("window")

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get frames per second (fps) of input video stream
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up output video
    output_file = "output/"+METHOD+"_kalman_"+VIDEO_FILE.split("/")[-1]
    if 2*w > 1920:
        out = cv2.VideoWriter(output_file, fourcc, fps, (1920, h))
    else:
        out = cv2.VideoWriter(output_file, fourcc, fps, (2*w, h))

    if not out.isOpened():
        print("Error: Could not open output video!")
        exit()


    # Read first frame
    _, prev = cap.read() 
    
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32) 
    
    start_time = time.time()
    sum_ace = 0
    for i in range(n_frames-2):
        # Read next frame
        success, curr = cap.read()
        print(METHOD," processing frame : ",i,"/",n_frames)

        if not success:
                break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        

        if METHOD =="OPTICAL_FLOW":
            # Detect feature points in previous frame
            prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                            maxCorners=200,
                                            qualityLevel=0.01,
                                            minDistance=30,
                                            blockSize=3)
                
            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
            
            # Sanity check
            assert prev_pts.shape == curr_pts.shape 
            
            # Filter only valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
        elif METHOD in ["SIFT","ORB","SURF"]:
            H,prev_features,curr_features,matched_image = get_features(prev_gray,curr_gray,METHOD)
            prev_pts = prev_features.matched_pts
            curr_pts = curr_features.matched_pts
            # Show the resulting image with matches
            # cv2.imshow("Matches", matched_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        elif METHOD=="SRHENET":        
            # Define the size of the patch to extract

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


        elif METHOD == "LOFTR":
            
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

            
        # print(prev_pts.shape)
        # print(prev_pts)
        #Find transformation matrix
        if METHOD == "LOFTR":
            m,_ =  cv2.estimateAffinePartial2D(prev_pts, curr_pts) #cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
        else :
            m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

        if m is None:
            print(METHOD, " couldnt found rigid transform, passing frame :", i)
            continue
        # Extract traslation
        #print(m)
        dx = m[0,2]
        dy = m[1,2]

        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])


        # Store transformation
        transforms[i] = [dx,dy,da]

        # Move to next frame
        prev_gray = curr_gray

        #print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(METHOD,"Elapsed time for computing trajectory:", elapsed_time, "seconds")
    print("mace : ",sum_ace/(n_frames-2))
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

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


    
    if smoothing_method == "kalman":
        smoothed_trajectory = smooth(trajectory, smoothing_radius=50)
    elif smoothing_method == "mavg":
        smoothed_trajectory = smooth_movingaverage(trajectory, smoothing_radius=50)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    
    # Calculate newer transformation array
    transforms_smooth = transforms + difference


    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break
    
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]
    
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
    
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h))
    
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 
    
        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])
        
        if(frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (1920, h))

        cv2.putText(frame_out, "runtime(s) :"+str(round(elapsed_time,3)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Before and After", frame_out)
        out.write(frame_out)
        cv2.waitKey(10)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

        
