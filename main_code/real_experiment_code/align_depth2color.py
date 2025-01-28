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

import pdb

def cam_setup():
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

    return pipeline, align, clipping_distance

def capture(pipeline, align, clipping_distance, img_id, save_addr=None):
    # Streaming loop
    try:
        while True:
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

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            #images = np.hstack((color_image, depth_image_3d))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('t'):
                if save_addr is None:
                    cv2.imwrite('/home/j0k/Project/Imsa/main_code/test_data/test_real_experiment/images/test_rgb_' + str(img_id) + '.jpg', color_image)
                    cv2.imwrite('/home/j0k/Project/Imsa/main_code/test_data/test_real_experiment/images/test_depth_' + str(img_id) + '.png', depth_image.astype(np.uint16))
                else:
                    cv2.imwrite(save_addr + 'test_image/' + str(img_id) + '.jpg', color_image)
                    cv2.imwrite(save_addr + 'test_depth_image/' + str(img_id) + '.png', depth_image.astype(np.uint16))
                    cv2.destroyAllWindows()

                return color_image, depth_image
    finally:
        pipeline.stop()

if __name__ == '__main__':
    img_id = int(sys.argv[1])
    pipeline, align, clipping_distance = cam_setup()
    capture(pipeline, align, clipping_distance, img_id)