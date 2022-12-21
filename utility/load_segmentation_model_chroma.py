import os
import sys
import json
from pathlib import Path
from typing import Any, TypedDict
import cv2
import torch
import numpy as np
from numpy.typing import NDArray

from os.path import join as pjoin

#base_path = os.path.dirname(os.path.abspath("."))
#sys.path.append(base_path)

from lib import chromakey
#from ove6d.lib import chromakey

def load(cfg, device, **kwargs)-> tuple[NDArray, torch.tensor, None]:
    """
    Helper function that loads a chromakey segmentation.
    
    Parameters
    ----------
    
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

    segmentator_setter = chromakey.Segmentator(
	        img_size=img_size
	        )
	
	#if object_name == 'test_gear':
	#    tola = 1.0; tolb = 1.53
	#elif object_name == 'test_clipper':
	#    tola = 0.66; tolb = 1.05
	#else:
	#    raise ValueError("What scene?")
    init_tola: int = 511
    init_tolb: int = 601
    init_Cb_key: int = 96
    init_Cr_key: int = 63
    pre_kwargs = {
	        'init_tola' : init_tola/10,
	        'init_tolb' : init_tolb/10,
	        'init_Cb_key' : init_Cb_key*1.0,
	        'init_Cr_key' : init_Cr_key*1.0
	        }

    mask_gpu = torch.zeros((1, img_size[0], img_size[1]), device=device, dtype=bool)
    mask: NDArray[np.bool_] = np.zeros((1, img_size[0], img_size[1]), dtype=np.bool_)

    #if kwargs['trackbars_on']:
    if True:
        window_name = 'mask'

        cv2.namedWindow(window_name)
        tola_max = 1000
        tolb_max = 1000
        Cb_key_max = 255
        Cb_key_max = 255

        def nothing(value) -> None:
            """
            Placeholder for OpenCV Trackbars
            Trackbars allow to tune the chromakey attributes.
            """
            pass

        # Create trackbars for color change and tolerance change
        # Trackbars are used to tune the chromakey attributes.
        # Only tola is in use at the moment. TODO Clean.
        cv2.createTrackbar('tola', window_name, init_tola, tola_max, nothing)
        cv2.createTrackbar('tolb', window_name, init_tolb, tolb_max, nothing)
        cv2.createTrackbar('Cb_key', window_name, init_Cb_key, Cb_key_max, nothing)
        cv2.createTrackbar('Cr_key', window_name, init_Cr_key, Cb_key_max, nothing)

        filter_ = segmentator_setter.get_filter(colorspace='YCrCb')

        # Margins in pixels
        w_margin = 10
        h_margin = 5
        margin_mask = np.zeros(img_size, dtype=np.uint8)
        margin_mask[:h_margin, :] = 1
        margin_mask[-h_margin:, :] = 1
        margin_mask[:, :w_margin] = 1
        margin_mask[:, -w_margin:] = 1


        def segmentator(image) -> tuple[NDArray[np.bool_], Any, None]:

            tola = cv2.getTrackbarPos('tola',window_name)/10
            tolb = cv2.getTrackbarPos('tolb',window_name)/10
            Cr_key = cv2.getTrackbarPos('Cr_key',window_name)
            Cb_key = cv2.getTrackbarPos('Cb_key',window_name)
            
            kwargs: dict[str, Any] = {"tola": tola, "tolb": tolb, "Cr_key": Cr_key, "Cb_key": Cb_key}
        	
            mask[:] = filter_(image=image, **kwargs)

            ### Get connected object component
            output = cv2.connectedComponentsWithStats(image=mask[0].astype(np.uint8), ltype=cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output

            # Define a small margin around the borders of the background.
            top_rows = labels[:h_margin, :]
            bot_rows = labels[-h_margin:, :]
            
            top_cols = labels[:, :w_margin]
            bot_cols = labels[:, -w_margin:]
            
            boundary_labels = np.unique([
                *top_rows.flatten(), *bot_rows.flatten(),
                *top_cols.flatten(), *bot_cols.flatten()])
            
            
            # Set area in margin to be background.
            # This takes care of the green canvas as its the largest connected component.
            # This takes care of a small table green screen, where borders might not be green.
            for boundary_label in boundary_labels:
                stats[boundary_label, cv2.CC_STAT_AREA] = 0
            
            # Now the object is tha lcc.
            object_label = np.argmax(stats[:, cv2.CC_STAT_AREA])
            
            #cc_mask = mask[0].copy()
            cc_mask = np.zeros_like(mask[0], dtype=np.uint8)
            cc_mask[labels == object_label] = 1
            cc_mask[cc_mask != 0] = 1
            

            mask[:] = cc_mask.astype(bool)[None,...]
            mask_gpu[:] = torch.from_numpy(mask)
            return mask, mask_gpu, None
    else:
        filter_ = segmentator_setter.get_filter(colorspace='YCrCb', **pre_kwargs)
    	
    	
        def segmentator(image) -> tuple[NDArray[np.bool_], Any, None]:
        	
            mask[:] = filter_(image=image)
            mask_gpu[:] = torch.from_numpy(mask)
            return mask, mask_gpu, None
    
    return segmentator
