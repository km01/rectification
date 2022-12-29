import cv2
import numpy as np
import visual_geometry


def reinforce_contours(src):
    src_gray = cv2.cvtColor(src, code=cv2.COLOR_BGR2GRAY)

    _, src_bin = cv2.threshold(src_gray, 
                            thresh=None, 
                            maxval=255, 
                            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return src_bin


def reorder_pts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]
    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]
    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts


def find_rectangles(src):
    
    candidates = []
    contours, _ = cv2.findContours(src, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    for pts in contours:    
        area = cv2.contourArea(pts)
        if area >= 1000:
            approx = cv2.approxPolyDP(pts, 0.02 * cv2.arcLength(pts, True), True)
            if len(approx) == 4:
                candidate = {'area' :area, 'pts': pts, 'approx': approx, 
                             'lu_lb_rb_ru': reorder_pts(approx.reshape(4, 2))}
                candidates += [candidate]
    return candidates


def get_biggest_rectangle(candidates):
    max_area = 0
    max_area_index = 0
    
    for i, cand in enumerate(candidates):
        if max_area < cand['area']:
            max_area_index = i
    
    return candidates[max_area_index]    
