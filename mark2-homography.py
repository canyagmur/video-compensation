import tkinter as tk
from tkinter import *
from tkinter import filedialog
import argparse

# Import numpy and OpenCV
import numpy as np
import cv2
from utils import *
import time


# Custom validation function for resize argument
def resize_range(value):
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("Resize value must be between 0 and 1")
    return value

METHODS =["OPTICAL_FLOW","SIFT","ORB","SURF"] #,"LOFTR","SRHENET"]
# Create the argument parser
parser = argparse.ArgumentParser(description='Video stabilization options')
# Add the method argument
parser.add_argument('-method', choices=METHODS, help='Specify the desired method for video stabilization', required=True)
# Add the smoothing argument
# Add the required window_size argument
parser.add_argument('-window_size', type=int, help='Window size for stabilizing the current frame', default=15)

# Add the required resize argument with custom validation
parser.add_argument('-resize', type=resize_range, help='Resize factor for frames (0-1)', default=1)

# Parse the arguments
args = parser.parse_args()


root = tk.Tk()
root.withdraw()
VIDEO_FILE = filedialog.askopenfilename()
root.destroy()

if len(VIDEO_FILE) == 0:
    print('please select a video.')
    quit()





METHOD = args.method
WINDOW_SIZE = args.window_size 
skip = 1 # speedup -- set 1 for original speed
assert skip ==1,"for now, no skip!" #TODO Need to change the code to support this, keep the indices of not skipped etc..
resize = args.resize #scale video resolution


# Set up output video
import os
output_folder = "output_homography"
os.makedirs(output_folder, exist_ok=True)
output_file = "{}/{}_w{}_r{}_{}".format(output_folder,METHOD, WINDOW_SIZE,resize,VIDEO_FILE.split("/")[-1])


# Use the values in your code
print(f"Method: {METHOD}")
print(f"Window Size : {WINDOW_SIZE}")
print(f"Resize ratio : {resize}")
print(f"Video file : {VIDEO_FILE}")


# if METHOD == "SRHENET":
#     model  = infer_srhen_model("C:/Users/PC_4232/Desktop/can/SRHEN-main/model_weights/srhen2/model_45.pt",device=DEVICE)
# elif METHOD == "LOFTR":
#     model = infer_loftr_model(pretrained_type="outdoor",device=DEVICE)

cap = cv2.VideoCapture(VIDEO_FILE)

frames = []
mean_homographies = []
median_homographies = []
corrected_frames = []
i = 0


while True:
    if cap.grab():
        flag, frame = cap.retrieve()
        if not flag:
            continue
        elif i%skip == 0:
            frame = cv2.resize(frame, (0,0), fx=resize, fy=resize)
            #cv2.imshow('video', frame)
            frames.append(frame)
        i+=1
    else:
        break
    #cv2.waitKey(1)

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#size_orig = (frames[0].shape[1], frames[0].shape[0])
#fps = cap.get(cv2.CAP_PROP_FPS)
#output_file = "output/"+METHOD+"_kalman_"+VIDEO_FILE.split("/")[-1]
# out_orig =  cv2.VideoWriter(VIDEO_FILE+'__original.avi',fourcc,fps,size_orig)#cv2.VideoWriter('stab.mp4',-1, 30.0, (frames[0].shape[0], frames[0].shape[1]))

# #write original video with skip x speedup
# for frame in frames:
#     out_orig.write(frame)

#out_orig.release()


start_time = time.time()

mean_homographies = []
median_homographies = []
for i in range(len(frames)):
    # Read next frame
    print(METHOD," processing frame : ",i+1,"/",len(frames))

    mean_H = np.zeros((3,3), dtype='float64')
    median_H = []
    mean_C = 0
    median_vals = []
    k =  int(WINDOW_SIZE/2.0)+1

    for j in range(1,k,1): #for each couple neighbor frames iterated by distance
        if i-j >= 0 and i+j < len(frames):
            prev_gray = cv2.cvtColor(frames[i-j], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            future_gray =  cv2.cvtColor(frames[i+j], cv2.COLOR_BGR2GRAY)



            if METHOD =="OPTICAL_FLOW":
                prev_pts, curr_pts , H = optical_flow_method(curr_gray,prev_gray)
                prev_pts2, curr_pts2 , H2 = optical_flow_method(prev_gray=curr_gray,curr_gray=future_gray)
            elif METHOD in ["SIFT","ORB","SURF"]:
                H,prev_features,curr_features,matched_image = get_features(curr_gray,prev_gray,METHOD)
                prev_pts = prev_features.matched_pts
                curr_pts = curr_features.matched_pts
                H2,prev_features2,curr_features2,matched_image2 = get_features(curr_gray,future_gray,METHOD)
                prev_pts2 = prev_features2.matched_pts
                curr_pts2 = curr_features2.matched_pts
            # elif METHOD =="SRHENET":
            #     prev_pts, curr_pts , H = srhenet_method(curr_gray,prev_gray,model=model)
            #     prev_pts2, curr_pts2 , H2 = srhenet_method(curr_gray,future_gray,model=model)
            # elif METHOD =="LOFTR":
            #     prev_pts, curr_pts , H = loftr_method(curr_gray,prev_gray,model=model)
            #     prev_pts2, curr_pts2 , H2 = loftr_method(curr_gray,future_gray,model=model)
    
            inliers_c = len(prev_pts)
            inliers_c2 = len(prev_pts2)
            print('pair (%d,%d) has %d inliers'% (i,i-j,inliers_c))
            print('pair (%d,%d) has %d inliers'% (i,i+j,inliers_c2))
            if inliers_c > 80 and inliers_c2 > 80: #ensures that neighbors are equally selected by distance to correctly balance the homography
                mean_H = mean_H + H
                mean_H = mean_H + H2
                mean_C+=2

    if mean_C > 0:
        mean_homographies.append(mean_H/mean_C) # Mean homography
    else:
        mean_homographies.append(np.eye(3, dtype='float64'))


end_time = time.time()
elapsed_time = end_time - start_time
print(METHOD,"Elapsed time for computing trajectory:", elapsed_time, "seconds")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

crop_x = 80
crop_y = 60

size = (frames[0].shape[1]-crop_x*2, frames[0].shape[0]-crop_y*2)
if 2*size[0] > 1920:
    out = cv2.VideoWriter(output_file, fourcc, fps, (1920, size[1]))
else:
    out = cv2.VideoWriter(output_file, fourcc, fps, (2*size[0], size[1]))

#out =  cv2.VideoWriter(output_file,fourcc,fps,size)#cv2.VideoWriter('stab.mp4',-1, 30.0, (frames[0].shape[0], frames[0].shape[1]))

for i in range(len(frames)):
    corrected = cv2.warpPerspective(frames[i],mean_homographies[i],(0,0))
    #cv2.imshow('video corrected', corrected)
    #cv2.waitKey(10)
    frame_cropped = frames[i][crop_y:frames[0].shape[0]-crop_y, crop_x:frames[0].shape[1]-crop_x]
    corrected_crop = corrected[crop_y:frames[0].shape[0]-crop_y, crop_x:frames[0].shape[1]-crop_x]
    frame_out = cv2.hconcat([frame_cropped, corrected_crop])
    if(frame_out.shape[1] > 1920):
        frame_out = cv2.resize(frame_out, (1920, size[1]))
    out.write(frame_out)

#print corrected.shape

cap.release()
out.release()
print("Output File created : ",output_file)