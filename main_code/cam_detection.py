## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import sys

from PIL import Image
from obj_detection import detection

import pdb

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# pdb.set_trace()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# dection model setup
det = detection(None)

# Start streaming
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()

preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
print('preset range:'+str(preset_range))
for i in range(int(preset_range.max)):
 visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
 print('%02d: %s'%(i,visulpreset))
 if visulpreset == "Default":
    depth_sensor.set_option(rs.option.visual_preset, i)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
#depth_sensor = profile.get_device().first_depth_sensor()
#color_sensor = profile.get_device().query_sensors()[1]
#color_sensor.set_option(rs.option.enable_auto_white_balance, False)
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 100 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# img_id = int(sys.argv[1])

profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
depth_intrinsics = depth_profile.get_intrinsics()
print (depth_intrinsics)

# Streaming loop
while True:
    img_id = input("Input view number: ")
    if img_id == 'q':
        break
    
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    cv2.imwrite('real_test_data/images/test_rgb_' + str(img_id) + '.jpg', color_image)
    cv2.imwrite('real_test_data/images/test_depth_' + str(img_id) + '.png', depth_image.astype(np.uint16))

    det = detection(color_image, text_promt=["keyboard", "pen"])
    masks = det.get_seg(mask_viz=True)


pipeline.stop()
