import tkinter as tk
from tkinter import *
from tkinter import filedialog
import argparse


# Import numpy and OpenCV
import numpy as np
import cv2
from utils import *
import time


METHODS =["OPTICAL_FLOW","SIFT","ORB","SURF"] #,"LOFTR","SRHENET"]
SMOOTHING_METHODS = ["mavg","kalman"] #kalman, mavg

# Create the argument parser
parser = argparse.ArgumentParser(description='Video stabilization options')
# Add the method argument
parser.add_argument('-method', choices=METHODS, help='Specify the desired method for video stabilization', required=True)
# Add the smoothing argument
parser.add_argument('-smoothing', choices=SMOOTHING_METHODS, help='Set the smoothing method', required=True)
# Parse the arguments
args = parser.parse_args()

root = tk.Tk()
root.withdraw()
VIDEO_FILE = filedialog.askopenfilename()
root.destroy()

if len(VIDEO_FILE) == 0:
	print('please select a video.')
	quit()

# import torch

# # Check if CUDA is available
# if torch.cuda.is_available():
#     DEVICE = "cuda"
# else:
#     DEVICE = "cpu"




# Access the method and smoothing values
METHOD = args.method
smoothing_method = args.smoothing

# Set up output video
import os
output_folder = "output_affine"
os.makedirs(output_folder, exist_ok=True)
output_file = "{}/{}_{}_{}".format(output_folder,METHOD, smoothing_method,VIDEO_FILE.split("/")[-1])


# Use the values in your code
print(f"Method: {METHOD}")
print(f"Smoothing: {smoothing_method}")
print(f"Video file : {VIDEO_FILE}")
# print(f"Device: {DEVICE}")

# if METHOD == "SRHENET":
#     model  = infer_srhen_model("C:/Users/PC_4232/Desktop/can/SRHEN-main/model_weights/srhen2/model_45.pt",device=DEVICE)
# elif METHOD == "LOFTR":
#     model = infer_loftr_model(pretrained_type="outdoor",device=DEVICE)

# Read input video
cap = cv2.VideoCapture(VIDEO_FILE)
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Get frames per second (fps) of input video stream
fps = cap.get(cv2.CAP_PROP_FPS)



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
for i in range(n_frames):
    # Read next frame
    success, curr = cap.read()
    print(METHOD," processing frame : ",i+1,"/",n_frames)

    if not success:
            break

    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    

    if METHOD =="OPTICAL_FLOW":
        prev_pts, curr_pts , _ = optical_flow_method(prev_gray=prev_gray,curr_gray=curr_gray)
    elif METHOD in ["SIFT","ORB","SURF"]:
        H,prev_features,curr_features,matched_image = get_features(prev_gray,curr_gray,METHOD)
        prev_pts = prev_features.matched_pts
        curr_pts = curr_features.matched_pts
        # Show the resulting image with matches
        # cv2.imshow("Matches", matched_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    # elif METHOD=="SRHENET":        
    #     # Define the size of the patch to extract
    #     prev_pts, curr_pts, _ = srhenet_method(prev_gray,curr_gray,model)

    # elif METHOD == "LOFTR":
    #     prev_pts, curr_pts, _ = loftr_method(prev_gray,curr_gray,model)
        

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
#print("mace : ",sum_ace/(n_frames-2))
# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)



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
print("Output File created : ",output_file)