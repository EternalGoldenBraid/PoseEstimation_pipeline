import os
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

from lib import rendering

class Dataset():
    def __init__(self, data_dir, cfg, 
            cam_K, cam_height, cam_width, n_triangles, object_renderer=None):
        self.model_dir = Path(data_dir) / 'models_eval'
        print("self.model_dir: ", self.model_dir)
        self.cam_file = Path(data_dir) / 'camera.json'
        print("self.cam_file: ", self.cam_file)
        self.cam_K = cam_K
        self.cam_K_np = cam_K.cpu().numpy()
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.model_info = None
        self.obj_model_file: dict = dict()
        self.obj_diameter = dict()
        self.point_cloud = dict()
        self.faces = dict()
        self.object_renderer = object_renderer
        self.cfg = cfg
        self.n_triangles: int = n_triangles # Number of points

        self.model_info_file = self.model_dir / 'models_info.json'
        #self.model_info_file = os.path.abspath(os.path.join(
            #os.path.dirname(__file__), '..', self.model_dir, 'models_info.json'))
        with open(self.model_info_file, 'r') as model_f:
            self.model_info = json.load(model_f)
        
        ### TODO: Combine faces and verts into same container.
        for model_file in sorted(self.model_dir.iterdir()):
            print("model_file: ", model_file)
            if str(model_file).endswith('.ply'):
                obj_id = int(model_file.name.split('_')[-1].split('.')[0])
                self.obj_model_file[obj_id] = model_file
                self.obj_diameter[obj_id] = self.model_info[str(obj_id)]['diameter']

                mesh = o3d.io.read_triangle_mesh(filename=str(model_file))
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n_triangles)
                self.point_cloud[obj_id] = np.asanyarray(mesh.vertices).astype(float)
                self.faces[obj_id] = np.asanyarray(mesh.triangles).astype(int)

        if self.cam_K == None:
            print("Warning camera intrinsics not set.")

        else:
            self.obj_renderer = rendering.Renderer(width=self.cfg.RENDER_WIDTH, height=self.cfg.RENDER_HEIGHT)
            #self.vis = o3d.visualization.Visualizer()
            #self.vis.create_window(visible=True)

        print("self.obj_model_file: ", self.obj_model_file)
    def render_cloud(self, obj_id, R, t, image):
        """
        Project the 3D points on to the image plane.
        """

        # FLIP Y, and Z coords
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t)
        
        P = P // P[-1,:] 
        # 检查深度值是否有效
        # if np.any(P[-1, :] <= 0):
        #     print(f"Error: Invalid depth values in P: {P[-1, :]}")
        #     return image, False

        # P = P / P[-1, :]


        ### TODO: Render just the points inside.
        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width or P[1].min() <= 0 or P[0].min() <= 0:
            return image, False

        P = P.astype(int)
        if np.any((P[1] < 0) | (P[1] >= image.shape[0]) | (P[0] < 0) | (P[0] >= image.shape[1])):
            print(f"Error: P is out of bounds: P={P}, image.shape={image.shape}")
            print("self.cam_height: ", self.cam_height)
            print("self.cam_width: ", self.cam_width)
            print(f"R: {R}, t: {t}, self.point_cloud[obj_id]: {self.point_cloud[obj_id]}")
            print(f"self.cam_K_np: {self.cam_K_np}")
        else:
            image[P[1], P[0], :] = 255

        return image, True

    def render_mesh(self, obj_id, R, t, image, color=(0,0,255), alpha=0.5):
        """
        Project the 3D points on to the image plane and color the faces with OpenCV fillpoly().
        """
        ### FLIP Y, and Z coords
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t).squeeze()
        P = P // P[-1,:]
        # 检查深度值是否有效
        # if np.any(P[-1, :] <= 0):
        #     print(f"Error: Invalid depth values in P: {P[-1, :]}")
        #     return image, False

        # P = P / P[-1, :]
        

        ### TODO: Render just the points inside.
        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            print("LOG: Projection out of bounds")
            return image, False

        P = P.astype(np.int32)
        #Does not fill the triangels since all are passed as a single batch.
        #Iterating over earch triangle and rendering one by one results in fill.
        image_filled = cv2.fillPoly(image.copy(), 
                                    #lineType=cv2.FILLED,
                                    #lineType=-1,
                                    pts=np.array(P.T[self.faces[obj_id]][:,:,:2]), color=color)

        image = cv2.addWeighted(image, alpha, image_filled, 1-alpha, 0.0)

        #image = cv2.drawContours(image=image, contourIdx=-1, color=color, thickness=-1, 
                #contours=np.array(P.T[self.faces[obj_id]][:,:,:2]))

        return image, True

    def render_mesh_slow(self, obj_id, R, t, image, color=(0,0,255), alpha=0.5):
        ### FLIP Y, and Z coords
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t).squeeze()
        P = P // P[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            print("LOG: Projection out of bounds")
            return image, False

        P = P.astype(np.int32)
        #Does not fill the triangels since all are passed as a single batch.
        #Iterating over earch triangle and rendering one by one results in fill.

        #for face_idx = 
        pts = np.array(P.T[self.faces[obj_id]][:,:,:2])
        image_filled = image.copy()
        for polygon in pts:
            #breakpoint()
            image_filled = cv2.fillPoly(image_filled,
                                        #lineType=cv2.FILLED,
                                        #lineType=-1,
                                        pts=[polygon], color=color)

        image = cv2.addWeighted(image, alpha, image_filled, 1-alpha, 0.0)

        #image = cv2.drawContours(image=image, contourIdx=-1, color=color, thickness=-1, 
                #contours=np.array(P.T[self.faces[obj_id]][:,:,:2]))

        return image, True
