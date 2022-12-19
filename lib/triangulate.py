import os
from scipy.spatial import Delaunay
import json
from pathlib import Path
from numba import njit
import numpy as np
import torch
from pytorch3d.io import load_ply
from numpy.random import default_rng
from ipdb import iex
import cv2

# Check if a point is inside a rectangle
def rect_contains(rect, point) :

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Draw a point
def draw_point(img, p, color ):
    cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ):
    triangleList = subdiv.getTriangleList();
    size = img.shape

    #import pdb; pdb.set_trace()
    r = (0, 0, size[1], size[0])
    for t in triangleList :
        t = t.astype(int).astype(np.uint8)
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)
            #cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
            #cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
            #cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)
