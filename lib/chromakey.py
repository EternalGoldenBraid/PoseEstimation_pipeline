import os
import threading
from pathlib import Path
import argparse

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Generator, List
from numpy.random import default_rng

from time import perf_counter

def draw_contours(image_size, mask: NDArray[np.uint8], color=(255,0,0), min_area=300, max_area=10000):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = np.array([cv2.contourArea(cnt) for cnt in contours])
    contours = [contours[i] for i in range(len(contours)) if area[i] > min_area and area[i] < max_area]

    new_mask = np.zeros([len(contours), img_size[0], img_size[1], 1], dtype=np.uint8)

    for c_idx in range(len(contours)):
        #new_mask[:,:,c_idx] = cv2.drawContours(mask, contours, c_idx, (1), -1)
        #breakpoint()
        #new_mask[c_idx] = cv2.fillPoly(new_mask[c_idx], pts =[contours[c_idx]], color=255)
        cv2.fillPoly(new_mask[c_idx], pts =[contours[c_idx]], color=255)
    return new_mask.squeeze(axis=-1).astype(bool)

def chroma_key_YCbCr(image: NDArray, tola: float, tolb: float, Cr_key: float, Cb_key: float) -> NDArray:
    """
    Chroma key with manually defined Key colors
    """

    arr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Cr: NDArray = arr[:,:,1].astype(float)
    Cb: NDArray = arr[:,:,2].astype(float)

    # Distance to key
    dist = np.sqrt((Cb_key-Cb)**2 + (Cr_key-Cr)**2)

    mask = np.ones_like(Cb, dtype=float)

    mask[dist < tola] = 0.0
    #mask[(dist < tolb) * (dist >= tola)] = (dist[(dist < tolb) * (dist >= tola)]-tola)/(tolb-tola)

    return mask

class Segmentator:

    def __init__(self, 
                 max_n_ref_frames: int =8,
                 img_size: tuple[int, int] = (480, 640),
                 seed=1337) -> None:
        self.rng = default_rng(seed)
        self.img_size: tuple[int, int] = img_size

    def get_filter(self, 
                hist_size: int = 256,
                hist_range: List[int] = [0,255],
                max_distance=1.5,
                **pre_kwargs) -> 'rgbd_2_mask':
        """
        Compute mask by chromakeying
        """

        #empty_mask = np.zeros((self.img_size[0], self.img_size[1]), dtype=bool)

        keys = pre_kwargs.keys()
        if 'init_tola' in keys and 'init_tolb' in keys and \
                'init_Cb_key' in keys and 'init_Cr_key' in keys:

            tola = pre_kwargs['init_tola']
            tolb = pre_kwargs['init_tolb']
            Cb_key = pre_kwargs['init_Cb_key']
            Cr_key = pre_kwargs['init_Cr_key']


            ### Return filter for fixed initial params.
            def filter_(image: NDArray[np.uint8], **kwargs)-> NDArray:
                """
                TODO
                """

                mask = chroma_key_YCbCr(image=image, 
                                  tola=tola, tolb=tolb,
                                  Cr_key=Cr_key, Cb_key=Cb_key
                                  )
                assert mask.size != 0

                #mask = cv2.blur(src=mask.astype(np.uint8), ksize=[3, 3])
                mask = cv2.morphologyEx(
                                            src=mask, 
                                            op=cv2.MORPH_CLOSE,
                                            #op=cv2.MORPH_DILATE,
                                            #op=cv2.MORPH_OPEN,
                                            kernel=np.ones(3, dtype=np.uint8),
                                            iterations=2)

                #mask = draw_contours(self.img_size, mask=mask.astype(np.uint8),
                                           #color=(255,0,0), min_area=min_area, max_area=max_area)
                return mask[None,...]

        else:

            ### Return filter for adjustable params.
            def filter_(image: NDArray[np.uint8], **kwargs)-> NDArray:
                """
                TODO
                """

                mask = chroma_key_YCbCr(image=image, 
                                  tola=kwargs['tola'], tolb=kwargs['tolb'],
                                  Cr_key=kwargs['Cr_key'], Cb_key=kwargs['Cb_key']
                                  )
                assert mask.size != 0

                #mask = cv2.blur(src=mask.astype(np.uint8), ksize=[3, 3])
                #mask = cv2.morphologyEx(src=mask, kernel=np.ones(5, dtype=np.uint8), iterations=4
                #            op=cv2.MORPH_CLOSE,
                #            #op=cv2.MORPH_DILATE,
                #            #op=cv2.MORPH_OPEN,
                #            )

                #return np.logical_not(mask.astype(bool)[None,...])
                return mask[None,...]
            
        return filter_

if __name__=='__main__':

    parser = argparse.ArgumentParser(description = 'Test segmentation')
    parser.add_argument('--scene', dest='scene', type=str, required=True, 
        choices = ['tabletop', 'green_canvas', 'test'], help="foo")
    parser.add_argument('--object', dest='object_name', type=str, required=True, 
        choices = ['tabletop', 'green_canvas', 
                   'test', 'test_box', 'test_gear', 'test_clipper', 'test_headphones', 'test_cup',
                   ], help="foo")
    parser.add_argument('--colorspace', dest='colorspace', type=str, required=True, 
        choices = ['rgb', 'YCbCr', 'lab', 
                   ], help="foo")
        
    args = parser.parse_args()

    ### TODO Load camera.
    ### TODO Make package for global path.

    img_size: tuple[int, int] = (640, 480)
    segmentator = Segmentator(img_size=img_size)

    cam_K = cam_K.astype(np.float64)
    frames = [(c, d) for (c, d) in zip(colors, depths)]
    n_frames = len(frames)
    print("Loaded", n_frames, "frames.")

    masked_color = np.zeros_like(colors[0])
    masked_depth = np.zeros_like(depths[0])

    window_name = 'win'
    def nothing(value) -> None:
        pass

    # create trackbars for color change
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.colorspace=='YCbCr' or args.colorspace =='lab':

        # a < b see function chroma_key
        #init_tola = 498 
        init_tola = 524
        init_tolb = 601
        tola_max = 1000
        tolb_max = 1000
        #tola_max = 255
        #tolb_max = 255

        ### RGB[0, 255, 0] in YCbCr
        #init_Cb_key = 44 
        #init_Cr_key = 22 

        ### Handtuned for RGB[50, 150, 100]
        #init_Cb_key = 111
        #init_Cr_key =  86

        ### Just works
        init_Cb_key = 96
        init_Cr_key = 63 

        Cb_key_max = 255
        Cr_key_max = 255 

        cv2.createTrackbar('tola', window_name, init_tola, tola_max, nothing)
        cv2.createTrackbar('tolb', window_name, init_tolb, tolb_max, nothing)
        cv2.createTrackbar('Cb_key', window_name, init_Cb_key, Cb_key_max, nothing)
        cv2.createTrackbar('Cr_key', window_name, init_Cr_key, Cb_key_max, nothing)

        pre_kwargs = {"init_tola": init_tola/10, "init_tolb": init_tolb/10, 
                      "init_Cr_key": init_Cr_key, "init_Cb_key": init_Cb_key}

    elif args.colorspace == 'rgb':

        # a < b see function chroma_key
        init_tola = 100
        init_tolb = 100
        tola_max = 1000
        tolb_max = 1000

        cv2.createTrackbar('tola', window_name, init_tola, tola_max, nothing)
        cv2.createTrackbar('tolb', window_name, init_tolb, tolb_max, nothing)

    else:
        lower_chl1_max = 255
        lower_chl2_max = 255
        lower_chl3_max = 255

        upper_chl1_max = 255
        upper_chl2_max = 255
        upper_chl3_max = 255

        lower = np.empty(3, dtype=np.uint8)
        upper = np.empty(3, dtype=np.uint8)

        init_min_area = 300
        init_max_area = 30000
        init_max_trials = 2
        init_eps = 9
        #init_eps = 60

        cv2.createTrackbar('min_area',window_name,init_min_area,5000,nothing)
        cv2.createTrackbar('max_area',window_name,init_max_area, 100000,nothing)
        cv2.createTrackbar('eps',window_name,init_eps,256,nothing)

    try:
        SAVE = False
        output_frames_buffer = np.empty([n_frames, img_size[0], img_size[1], 3], dtype=np.uint8)
        idx = -1
        kwargs = {}
        #filter_ = segmentator.get_filter(colorspace=args.colorspace, **pre_kwargs)
        filter_ = segmentator.get_filter(colorspace=args.colorspace)

        color_mask = np.concatenate((
            np.ones((img_size[0], img_size[1],1))*255,
            np.ones((img_size[0], img_size[1],1))*0,
            np.ones((img_size[0], img_size[1],1))*0,
            ),axis=2)
        
        while True:
            idx += 1

            c, d = frames[idx % n_frames]

            time_a = perf_counter()

            # get current positions of four trackbars
            if args.colorspace == 'YCbCr':
                 
                tola = cv2.getTrackbarPos('tola',window_name)/10
                tolb = cv2.getTrackbarPos('tolb',window_name)/10
                #tola = cv2.getTrackbarPos('tola',window_name)
                #tolb = cv2.getTrackbarPos('tolb',window_name)
                Cr_key = cv2.getTrackbarPos('Cr_key',window_name)
                Cb_key = cv2.getTrackbarPos('Cb_key',window_name)

                kwargs = {"tola": tola, "tolb": tolb, "Cr_key": Cr_key, "Cb_key": Cb_key}
            elif args.colorspace == 'rgb':
                 
                tola = cv2.getTrackbarPos('tola',window_name)/100
                tolb = cv2.getTrackbarPos('tolb',window_name)/100

                kwargs = {"tola": tola, "tolb": tolb}
            
            else:
                lower_chl1 = cv2.getTrackbarPos('lower_chl1',window_name)
                lower_chl2 = cv2.getTrackbarPos('lower_chl2',window_name)
                lower_chl3 = cv2.getTrackbarPos('lower_chl3',window_name)
                upper_chl1 = cv2.getTrackbarPos('upper_chl1',window_name)
                upper_chl2 = cv2.getTrackbarPos('upper_chl2',window_name)
                upper_chl3 = cv2.getTrackbarPos('upper_chl3',window_name)

                lower[:] = [lower_chl1, lower_chl2, lower_chl3]
                upper[:] = [upper_chl1, upper_chl2, upper_chl3]
                kwargs = {'lower':lower, 'upper': upper}


            mask = filter_(
                    image=c, depth=d*depth_scale,
                    #eps = eps, 
                    #max_residual=max_residual,
                    #min_area=min_area, max_area=max_area,
                    #max_distance=max_dist,
                    **kwargs
                    )

            processing_fps = 1/(perf_counter() - time_a)

            for item_idx in range(mask.shape[0]):
                #masked_color[mask[item_idx]] = (255,105,180)
                #masked_color[mask[item_idx]] = mask[item_idx]
                masked_color = (mask[item_idx][...,None].repeat(repeats=3, axis=-1)*c).astype(np.uint8)

                if args.colorspace=='rgb':
                    ### Deal with greenspill

                    spill_corrected = masked_color.copy()
                    spill_corrected[:,:,1] = np.minimum(masked_color[:,:,1], masked_color[:,:,0])
                    images = np.hstack((
                        masked_color,
                        spill_corrected,
                        (mask[item_idx][...,None].repeat(repeats=3, axis=-1)*255).astype(np.uint8)
                        ))
                else:
                    images = np.hstack((
                        masked_color,
                        (mask[item_idx][...,None].repeat(repeats=3, axis=-1)*255).astype(np.uint8)
                        ))


            cv2.putText(images, f"fps: {processing_fps:2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, images)
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

            if SAVE and idx < n_frames:
                # video.write(images)
                output_frames_buffer[idx] = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    #except IndexError as e:
        #print(e)
        #breakpoint()
    finally:
        import mediapy as media
        if SAVE:
            data_dir = Path('assets','histogram_reference',args.scene, args.plane_fit_to)
            data_dir.mkdir(exist_ok=True, parents=True)
            save_path = data_dir/object_name
            
            media.write_video(save_path.with_suffix('.mp5'), output_frames_buffer, fps=fps)
