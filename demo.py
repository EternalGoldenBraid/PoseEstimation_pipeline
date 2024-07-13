import os
from time import perf_counter
import warnings
from pathlib import Path
import cv2
import torch
import numpy as np
from OpenGL import EGL

warnings.filterwarnings("ignore")

#breakpoint()
#base_path = os.path.dirname(os.path.abspath("."))
#sys.path.append(base_path)
#
#breakpoint()
from utility import ( cam_control,
                     #load_segmentation_model,
                     load_segmentation_model_chroma, io_utils)
from utility.load_pose_estimator import PoseEstimator

from dataset import demo_dataset
from configs import config as cfg

DEVICE = torch.device('cuda')
# DEVICE = torch.device('cpu')
# DEVICE = 'cpu'
# DEVICE = 'cuda'

def initialize_egl():
    display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    if display == EGL.EGL_NO_DISPLAY:
        raise Exception("EGL not available")

    initialized = EGL.eglInitialize(display, None, None)
    if not initialized:
        raise Exception("Failed to initialize EGL")

    config_attributes = [
        EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
        EGL.EGL_RED_SIZE, 8,
        EGL.EGL_GREEN_SIZE, 8,
        EGL.EGL_BLUE_SIZE, 8,
        EGL.EGL_DEPTH_SIZE, 24,
        EGL.EGL_NONE
    ]

    num_configs = EGL.EGLint()
    config = (EGL.EGLConfig * 1)()
    if not EGL.eglChooseConfig(display, config_attributes, config, 1, num_configs):
        raise Exception("Failed to choose EGL config")

    context_attributes = [
        EGL.EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL.EGL_NONE
    ]

    context = EGL.eglCreateContext(display, config[0], EGL.EGL_NO_CONTEXT, context_attributes)
    if context == EGL.EGL_NO_CONTEXT:
        raise Exception("Failed to create EGL context")

    pbuffer_attributes = [
        EGL.EGL_WIDTH, 256,
        EGL.EGL_HEIGHT, 256,
        EGL.EGL_NONE,
    ]

    surface = EGL.eglCreatePbufferSurface(display, config[0], pbuffer_attributes)
    if surface == EGL.EGL_NO_SURFACE:
        raise Exception("Failed to create EGL Pbuffer surface")

    if not EGL.eglMakeCurrent(display, surface, surface, context):
        raise Exception("Failed to make EGL context current")

    return display, surface, context

def main(args):
    # Initialize EGL
    try:
        display, surface, context = initialize_egl()
        print("EGL initialized successfully")
    except Exception as e:
        print(f"EGL initialization failed: {e}")
        return
    cfg.DATASET_NAME = args.dataset_name
    cfg.USE_ICP = args.icp

    obj_id: int = io_utils.get_obj_id(
            dataset_path=Path(cfg.DATA_PATH, cfg.DATASET_NAME),
            name=args.object_name)

    # Load camera module
   # TODO change to row_major for numpy...? What's torch
    cam = cam_control.Camera(size=(cfg.RENDER_WIDTH, cfg.RENDER_HEIGHT), framerate=60)
    depth_scale, cam_K =  cam.depth_scale, cam.cam_K

    # load segmentation module
    if args.segment_method == 'chromakey':
        segmentator = load_segmentation_model_chroma.load(cfg=cfg, device=DEVICE)
    else:
        raise NotImplementedError("Enable segmentator selection (detectron)")
        #segmentator = load_segmentation_model.load(model=args.segment_method, cfg=cfg, device=DEVICE)

    # Load mesh models
    dataroot: Path = Path(os.path.dirname(__file__))/Path(cfg.DATA_PATH)
    #dataroot = Path(os.path.realpath(__file__)).parent.parent/Path(cfg.DATA_PATH)
    dataset = demo_dataset.Dataset(data_dir=dataroot/ cfg.DATASET_NAME, cfg=cfg,
                cam_K=cam_K, cam_height=cfg.RENDER_HEIGHT, cam_width=cfg.RENDER_WIDTH,
                n_triangles=args.n_triangles)

    # Load pose estimation module
    codebook_path: Path = dataroot/'object_codebooks'/ cfg.DATASET_NAME / \
        'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR) / \
        'views_{}'.format(str(cfg.RENDER_NUM_VIEWS))
    print("codebook_path: ", codebook_path)
    pose_estimator = PoseEstimator(cfg=cfg, cam_K=dataset.cam_K, obj_id=obj_id,
            model_path=Path('checkpoints','OVE6D_pose_model.pth'),
            device=DEVICE, dataset=dataset, codebook_path=codebook_path)

    d_max = 1 # Max depth range
    while True:

        fps_start = perf_counter()
        
        depth_image, color_image = cam.get_image()

        depth_image[depth_image*depth_scale > d_max] = 0
        depth_image[depth_image*depth_scale <= 0] = 0

        masks, masks_gpu, _ = segmentator(color_image)

        cv2.imshow('mask', masks[0].astype(float))

        #depth_colormap = cv2.applyColorMap(
        #        cv2.convertScaleAbs(depth_image,
        #                            #alpha=0.03),
        #                            alpha=0.9, beta=0.0),
        #        cv2.COLORMAP_JET) 
        #depth_colormap = (255*((depth_image - d_max)/d_max)).astype(np.uint8)[..., None].repeat(repeats=3,axis=2)
        #depth_colormap = cv2.applyColorMap(
        #        (255*((depth_image - d_max)/d_max)).astype(np.uint8),
        #        cv2.COLORMAP_JET) 
        depth_colormap = cv2.applyColorMap(
                (255*depth_image/depth_image.max() ).astype(np.uint8),
                cv2.COLORMAP_JET) 
        if masks.size != 0:

            ### TODO: Can we get depth_image dircetly to gpu from sensor and skip gpu --> cpu with <mask>
            R, t = pose_estimator.estimate_pose(obj_mask=masks_gpu[0][None,...],
                        obj_depth=torch.tensor(
                            (depth_image*masks[0]*depth_scale).astype(np.float32)).squeeze()[None,...]
                        )
            print("R: ", R)
            print("R.shape: ", R.shape)
            print("t: ", t)
            print("t.shape: ", t.shape)
            ### TODO Multi object support.
            #obj_depths = torch.tensor([(depth_image*mask*depth_scale).astype(np.float32) for mask in masks])
            #R, t = pose_estimator.estimate_poses(obj_masks=masks_gpu, scores=scores,
            #            obj_depths=obj_depths.squeeze())

            for transform_idx in range(R.shape[0]):

                if args.render_mesh:
                    color_image, done = dataset.render_mesh(obj_id=obj_id, 
                         R=R[transform_idx].numpy().astype(np.float32), 
                         t=t[transform_idx].numpy()[...,None].astype(np.float32),
                         image=color_image.copy())
                else:
                    color_image, done = dataset.render_cloud(obj_id=obj_id, 
                        R=R[transform_idx].numpy().astype(np.float32), 
                        t=t[transform_idx].numpy()[...,None].astype(np.float32),
                        image=color_image)


            images = np.hstack([ 
                color_image, 
                depth_colormap*np.array(masks.sum(axis=0, dtype=np.uint8)[...,None]) 
                #color_image*np.array(masks.sum(axis=0, dtype=np.uint8)[...,None]) 
                ])
        else:
            images = np.hstack((color_image, depth_colormap))

        
        cv2.putText(images, f"fps: {(1/(perf_counter()-fps_start)):2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


    del cam
    # Clean up EGL resources
    EGL.eglDestroyContext(display, context)
    EGL.eglDestroySurface(display, surface)
    EGL.eglTerminate(display)


if __name__=="__main__":
    import argparse

    from enum import Enum, unique
    class ArgTypeMixin(Enum):

        @classmethod
        def argtype(cls, s: str) -> Enum:
            try:
                return cls[s]
            except KeyError:
                raise argparse.ArgumentTypeError(
                    f"{s!r} is not a valid {cls.__name__}")

        def __str__(self):
            return self.name

    @unique
    class ObjectIds(ArgTypeMixin, Enum):
        """
        Contains all 3D model ids for this dataset.
        Add new objects below.
        """

    parser = argparse.ArgumentParser(prog='demo',
            description='Superimpose rotated pointcloud onto video.')

    parser.add_argument('-o','--object_name', dest='object_name', type=str,
                        default='box', #box
        help='Object name as specified in Dataspace/<dataset_name>/models_info.json')
    parser.add_argument('-n', '--n_triangles', dest='n_triangles',
                        type=int, required=False, default=2000,
                        help='Number of triangles for cloud/mesh.')
    parser.add_argument('-s', '--segmentation', dest='segment_method',
                        required=False, default='chromakey',
                        choices = ['chromakey','bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn', 'point_rend'],
                        help="""Method of segmentation.
                        contour: OpenCV based edge detection ...,
                        TODO:
                        """)
    parser.add_argument('-rm', '--render-mesh', dest='render_mesh', action=argparse.BooleanOptionalAction)
    parser.add_argument('-icp', dest='icp', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dataset_name', dest='dataset_name', default='demo_dataset', #demo_dataset
                        help="""Dataset name in path Dataspace/<dataset_name>/...""")
                        
    
    args = parser.parse_args()

    main(args)
