import imageio
import os

def create_gif(video_path, output_dir, output_filename, start_frame, end_frame):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the video using imageio
    video_reader = imageio.get_reader(video_path)

    # Create a GIF writer
    gif_writer = imageio.get_writer(os.path.join(output_dir, output_filename), mode='I')

    # Iterate over each frame in the specified range
    for frame_idx in range(start_frame, min(end_frame, len(video_reader))):
        # Read the frame
        frame = video_reader.get_data(frame_idx)

        # Write the frame to the GIF file
        gif_writer.append_data(frame)

    # Close the GIF writer and video reader
    gif_writer.close()
    video_reader.close()

# Example usage
video_file = "ORB_hm_drone2"
video_path = os.path.join('C:/Users/PC_4232/Desktop/can/video-compensation/examples/output_homography/',video_file+".mp4")
output_dir = r'C:\Users\PC_4232\Desktop\can\video-compensation\examples\output_affine\gifs'
output_filename = '{}.gif'.format(video_file)


start_frame = 100  # Specify the starting frame index or time
end_frame = 200    # Specify the ending frame index or time

create_gif(video_path, output_dir, output_filename,start_frame, end_frame)