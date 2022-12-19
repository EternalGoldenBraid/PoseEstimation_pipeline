import os
import sys
import json
from pathlib import Path
import cv2
import torch
import numpy as np
from numpy.typing import NDArray

from ipdb import iex

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

def load(model: str, cfg, device: str):
    """
    Helper function that loads a segmentation module defined.

    Parameters
    ----------

    :param string model: name of the segmentation module to be loaded.
    :param string device: cuda or cpu.
    :param cfg: a .py file defining various configuration parameters as variables.


    Returns
    -------

    :return torch.tensor mask_gpu: torch.tensor on gpu of shape [N_objects, img_height, img_width]
    :return numpy.array mask: np.array on cpu [N_objects, img_height, img_width]
    :return scores: Confidence scores if provided my segmentation model.


    Notes
    -----

    Place the segmentation model that defines that implements the described return types to ./lib
    from which it will be imported.
    """
    img_size=(cfg.RENDER_HEIGHT, cfg.RENDER_WIDTH)

    if model == 'your_segmentator_name':
        #from lib import your_segmentator

        def segmentator_(image):
            mask = cv2.createBackgroundSubtractorKNN().apply(image)
            return mask, torch.tensor(mask, device=device), None
    elif model == 'contour':
        from lib import contour_segmentation

        none_array = np.array(())
        model = contour_segmentation.BackgroundContour()

        def segmentator_(image):
            #mask = contour_segmentation.ContourSegmentator().get_mask(image)//255
            masks = model.get_mask(image)
            if len(masks) == 0:
                return none_array, None, None
            return masks, torch.tensor(masks, device=device), None

    elif model == 'maskrcnn':
        from lib import detectron_segmentation
        import detectron2.data.transforms as T
        from detectron2 import model_zoo

        none_array = np.array(())
        model_seg, model_cfg = detectron_segmentation.load_model_image_agnostic(
                base_path+'/checkpoints/FAT_trained_Ml2R_bin_fine_tuned.pth',
                device=device)

        aug = T.ResizeShortestEdge(
            [model_cfg.INPUT.MIN_SIZE_TEST, model_cfg.INPUT.MIN_SIZE_TEST], model_cfg.INPUT.MAX_SIZE_TEST
        )

        def segmentator_(image, model=model_seg, aug=aug):
            image = aug.get_transform(image).apply_image(image)
            inputs = {"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)),
                        "height": cfg.RENDER_HEIGHT, "width": cfg.RENDER_WIDTH}
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                pred = model([inputs])[0]['instances']
                mask_gpu = pred.get('pred_masks')
                scores = pred.get('scores')
                if mask_gpu.numel() == 0:
                    return none_array, None, None
                mask_cpu = mask_gpu.to(
                        non_blocking=True, copy=True, device='cpu').numpy().astype(int).astype(np.uint8)
                return mask_cpu, mask_gpu, scores

    elif model == 'point_rend':
        from detectron2.projects import point_rend
        from lib import detectron_segmentation
        from detectron2 import model_zoo

        none_array = np.array(())
        print("BASE:",base_path)
        #model_file = base_path+'/checkpoints/model_final_ba17b9_pointrend.pkl'
        model_path = base_path+'/checkpoints/model_final_edd263_pointrend.pkl'
        
        model_seg = detectron_segmentation.load_model_point_rend(model_path=model_path,
                config_yaml=base_path+ \
                        '/configs/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
                #config_yaml='configs/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
                confidence=0.7, base_path=base_path, device=device)
        def segmentator_(image, model=model_seg):
            pred = model(image)['instances']
            if pred.pred_masks.numel() == 0:
                return none_array, None, None
            #import pdb; pdb.set_trace()
            mask_gpu = pred.get('pred_masks')
            scores = pred.get('scores')
            mask_cpu = mask_gpu.to(
                    non_blocking=True, copy=True, device='cpu').numpy().astype(int).astype(np.uint8)
            return mask_cpu, mask_gpu, scores
    else: 
        raise NotImplementedError(f"Segmentation mode {model}, does not exist.")

    return segmentator_
