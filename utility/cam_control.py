import os
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

class Camera():
    """
    A camera module that provides color and depth stream. 
    Camera intrinsics accessible with self.cam_K.
    """

    def __init__(self, size=(640, 480), framerate=60):

        self.pipeline = None
        self.config = None
        self.align = None

        # Create a pipeline
        self.pipeline = rs.pipeline()
        
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
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
        
        self.config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, framerate)
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, framerate)
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        #align_to = rs.stream.depth
        self.align = rs.align(align_to)
    

        ### Get scale intrinsics
        ### TODO: Get directly from device without capturing frame.

        # Start streaming
        profile = self.pipeline.start(self.config)
    
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)
    
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
    
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            print("Couldn't load camera frame.")
            return
    
        i = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        #color_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics ### These should be same.
        self.cam_K = torch.tensor([  [i.fx,  0,  i.ppx],
                            [0, i.fy, i.ppy],
                            [0, 0,  1]], device='cpu') 
    
        print("Camera intrinsics:", self.cam_K)

        ## We will be removing the background of objects more than
        ##  clipping_distance_in_meters meters away
        #clipping_distance_in_meters = 1 #1 meter
        #clipping_distance = clipping_distance_in_meters / depth_scale

    
    def get_rs_frame(self):

        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None
        
        return aligned_depth_frame, color_frame

    def get_image(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def __del__(self):
            self.pipeline.stop()

    
if __name__=="__main__":

    cam = Camera(size=(640, 480), framerate=60)
    depth_scale, cam_K =  cam.depth_scale, cam.cam_K

    try:
        while True:

            depth_image, color_image = cam.get_image()

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 

            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        del cam

