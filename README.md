# Video Stabilization & Compensation

Video stabilization is a technique used to reduce unwanted motion and shaking in videos, resulting in smoother footage. It addresses issues caused by camera shake, hand movements, or unstable platforms during video recording. Techniques such as motion-based stabilization estimate the camera's motion trajectory and apply motion compensation to align frames and reduce shake. Optical flow-based stabilization analyzes pixel motion between frames to determine motion vectors for stabilization. Advanced methods utilize computer vision and machine learning algorithms to analyze motion patterns and improve stabilization accuracy. Multi-frame fusion techniques fuse information from multiple frames to further enhance stability. Video stabilization enhances the viewing experience, eliminates distractions, and improves overall video quality, benefiting applications like filmmaking, surveillance, and action cameras. Continuous advancements in technology drive the development of more effective video stabilization solutions.
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Create a Conda environment with the specified name and Python version 3.6.9:

   ```bash
   conda create -n video-stabilization python=3.6.9
   conda activate video-stabilization 
   pip install -r requirements.txt
    ```

This command will automatically install all the necessary packages and libraries required by the project.

**Note:** Make sure you have navigated to the project directory containing the `requirements.txt` file before running the above command.


## Usage

### Method 1 : Tracking the trajectory of extracted keypoints using specified method and smoothing the trajectory using specified filter
```
python mark1_affine.py -method <method_name> -smoothing <smoothing_method>

Available methods : "OPTICAL_FLOW","SIFT","ORB","SURF"
Available Smoothing methods : 'mavg', 'kalman'
```

### Method 2 : Calculating the  stabilized homography of the current frame with respect to previous frame by looking to window of frames with specified window size.

```
python mark2_homography.py -method <method_name>

Available methods : "OPTICAL_FLOW","SIFT","ORB","SURF"
```


## Background of Method 1
The logic of video stabilization can be summarized as follows:

1. Read the input video file and obtain its properties such as number of frames, width, height, and frames per second.

2. Initialize a transformation store array to store the transformations (translation and rotation) for each frame.

3. Iterate through each frame in the video:

   - Read the next frame.
   - Extract feature points from the current and previous frames using a selected method (optical flow or feature matching).
   - Estimate the transformation matrix (translation and rotation) between the current and previous frames.
   - Store the obtained transformation in the transformation store array.
   - Set the current frame as the previous frame for the next iteration.

4. Compute the trajectory by taking the cumulative sum of the transformations stored in the transformation store array.

5. Smooth the trajectory using a selected smoothing method (Kalman filter or moving average).

6. Calculate the difference between the smoothed trajectory and the original trajectory.

7. Generate a new transformation array by adding the difference to the original transformations.

8. Reset the video stream to the first frame.

9. Iterate through each frame (except the first and last) to apply the stabilized transformations:

   - Read the next frame.
   - Extract the transformation parameters (translation and rotation) from the new transformation array.
   - Construct an affine transformation matrix using the transformation parameters.
   - Apply the affine transformation to the current frame.
   - Fix any border artifacts caused by the transformation.
   - Concatenate the original frame and the stabilized frame horizontally.

10. Write the stabilized frames to an output video file.

The main idea behind video stabilization is to analyze the motion between consecutive frames and apply appropriate transformations to compensate for this motion. The stabilization process involves estimating the transformation between frames, smoothing the estimated trajectory, and applying the stabilized transformations to the frames. This helps in reducing unwanted motion and jitter in the video, resulting in a smoother and more stable output.

![ORB + Moving Average](C:/Users/PC_4232/Desktop/can/video-compensation/examples/output_affine/gifs/ORB_drone2.gif)
![ORB + Kalman](C:/Users/PC_4232/Desktop/can/video-compensation/examples/output_affine/gifs/ORB_kalman_drone2.gif)

## Background of Method 2

The code performs video stabilization using the following logic steps:

1. Import the necessary libraries.
2. Define the available stabilization methods.
3. Prompt the user to select a video file.
4. If a video file is selected, proceed; otherwise, display an error message and quit.
5. Print the selected video file, stabilization method, window size, and resize factor.
6. Set up the output video file and directory.
7. Open the video file using `cv2.VideoCapture`.
8. Read and store the frames of the video, possibly applying a resize factor if specified.
9. Initialize the mean and median homographies as empty lists.
10. For each frame in the list, perform the stabilization process as follows:
    - Compute the homography between the current frame and its neighboring frames using the selected method.
    - Calculate the inliers (matching points) between the frames.
    - If the number of inliers is above a threshold, add the homography to the mean homography and increment the count.
    - After processing all the neighboring frames, divide the mean homography by the count to obtain the average homography.
    - Append the mean homography to the mean homographies list.
11. Print the elapsed time for computing the trajectory.
12. Crop and concatenate the stabilized frames with the original frames.
13. Write the stabilized video to the output file.

![ORB + Moving Average](C:\Users\PC_4232\Desktop\can\video-compensation\examples\output_homography\gifs\ORB_hm_drone2.gif)

## Contributing

If you'd like to contribute to this video stabilization code, consider the following areas of interest:

- Implementing an online version: Adapt the code to perform real-time video stabilization as frames are captured, updating the stabilization parameters dynamically.

- Applying deep learning methods: Explore the possibility of leveraging deep learning techniques for video stabilization, allowing the model to learn complex motion patterns and improve stability.

Contributions in these areas would enhance the video stabilization capabilities of the code and explore new avenues for improvement.

## References
### Github repositiries that I utilized and took code from:
- https://github.com/xiyori/video_smoothing
- https://github.com/2vin/video-stabilization
- https://github.com/guipotje/homog_stab
- https://github.com/ll-nick/video-stabilizer
- https://github.com/Lakshya-Kejriwal/Real-Time-Video-Stabilization

### Papers 
- Video stabilization: A comprehensive survey, https://doi.org/10.1016/j.neucom.2022.10.008

### Other 
- https://stackoverflow.com/questions/3431434/video-stabilization-with-opencv?rq=4
- https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

