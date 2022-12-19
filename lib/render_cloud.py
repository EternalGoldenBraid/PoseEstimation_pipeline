import json
import torch
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor)

DEVICE = 'cuda'
#DEVICE = 'cpu'

#OBJ_IDX=args.object_idx
#SHOT_IDX=1
#obj_dir = data_path+'/test/{:06d}'.format(obj_idx)
#files = [f[1:-4].split('_')[1] for f in os.listdir(obj_dir+'/depth')]

def load_cloud(obj_idx=2, data_path='huawei_box', device='cpu'):

    ply_path = (data_path/'models_eval'/f'obj_00000{str(obj_idx)}.ply')
    #import pdb; pdb.set_trace()

    cam_info_file = (data_path/'camera.json')
    with open(cam_info_file, 'r') as cam_f:
        camera_info = json.load(cam_f)

    view_camK = torch.tensor([
        camera_info['fx'], 0, camera_info['cx'], 0, camera_info['fy'], camera_info['cy'], 0 , 0, 1
        ], dtype=torch.float32).view(3, 3)
    cam_K = view_camK

    #import pdb; pdb.set_trace()
    #obj_info_file = (data_path+'/models_eval/')

    ## Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
    #sphere_mesh = ico_sphere(level=3)
    verts, _ = load_ply(ply_path)
    #verts = verts.to(device)
    return verts, cam_K

def render_cloud(point_cloud, cam_K, R, t, image):
    #import pdb; pdb.set_trace()
    ### FLIP Y, and Z coords
    R = R@torch.tensor([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]], device='cpu', dtype=torch.float32)

    t = t.expand(point_cloud.shape[0],-1)

    P = (cam_K@(R@point_cloud.T + t.T))
    P = P / P[-1,:]
    P = P.int()
    #view_depth *= view_cam_info['depth_scale']
    #view_depth *= cfg.MODEL_SCALING # convert to meter scale from millimeter scale
    #view_depth/=view_depth.max()
    P = P.cpu().numpy()

    #import pdb; pdb.set_trace()
    #image[P[1], P[0], :] = P[2].expand(-1, 3)
    image[P[1], P[0], :] = 255 
    #import pdb; pdb.set_trace()
    #return view_depth

def render_cloud_old(point_cloud, cam_K, R, t):

    if OBJ_IDX == 1:
        pass
    elif OBJ_IDX == 2:
        ## BASKET
        R = torch.tensor([[-0.6845, -0.7180,  0.1258],
                        [ 0.6571, -0.5331,  0.5329],
                        [-0.3156,  0.4475,  0.8368]], device=DEVICE)
        t = torch.tensor([-0.0933,  0.0789,  0.3789], device=DEVICE)
    else:
        ## Headphones
        R = torch.tensor([[-0.4433, -0.8919,  0.0897],
                        [ 0.2298, -0.0164,  0.9731],
                        [-0.8664,  0.4520,  0.2123]], device=DEVICE)
        t = torch.tensor([0.0130, 0.0459, 0.2688], device=DEVICE)

    ### FLIP Y, and Z coords
    R = R@torch.tensor([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]], device=DEVICE, dtype=torch.float32)

    t = t.expand(point_cloud.shape[0],t.shape[0])

    P = (cam_K@(R@point_cloud.T + t.T))
    P = P / P[-1,:]
    P = P.int()

    #c = 'autumn'
    c = 'viridis'
    cmap = plt.get_cmap(c)

    depth_file = OBJ_DIR+'/depth/'+'_Depth_'+FILES[SHOT_IDX]+'.csv'
    #view_depth = torch.tensor(np.loadtxt(depth_file, delimiter=",", dtype=float), dtype=torch.float32) # In meters
    view_depth = np.loadtxt(depth_file, delimiter=",", dtype=float) # In meters
    #view_depth *= view_cam_info['depth_scale']
    #view_depth *= cfg.MODEL_SCALING # convert to meter scale from millimeter scale
    view_depth/=view_depth.max()
    P = P.cpu().numpy()

    view_depth[P[1], P[0]] = P[2]
    #view_depth[P[0], P[1]] = P[2]
    #plt.imshow(cmap(view_depth))
    #plt.show()
    im = view_depth

    plt.imshow(cmap(im))
    plt.show()

