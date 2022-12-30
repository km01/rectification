import os
import cv2
from PIL import Image
import numpy as np
import argparse
import visual_geometry
import rectangle_detact


def load_image(file_dir):
    img = Image.open(file_dir)
    img = np.array(img)
    return img

def save_image(img, file_dir):
    img = Image.fromarray(img)
    img.save(file_dir)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rectification')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--debug', type=bool, default=False)
    
    args = parser.parse_args()
    
    img = load_image(args.input_dir)
    preprocessed = rectangle_detact.reinforce_contours(img)
    
    candidates = rectangle_detact.find_rectangles(preprocessed)
    
    
    biggest_cand = rectangle_detact.get_biggest_rectangle(candidates)
    homograph, width_height = visual_geometry.rectification(biggest_cand['lu_lb_rb_ru'])

    rectified_img = cv2.warpPerspective(img, homograph, width_height)

    ratio = visual_geometry.compute_aspect_ratio(img, biggest_cand['lu_lb_rb_ru'])

    rectified_img = cv2.resize(rectified_img, dsize=(int(ratio * rectified_img.shape[0]), rectified_img.shape[0]))


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    save_image(rectified_img, file_dir=args.output_dir + '/result.jpg')

    if args.debug:

        if not os.path.exists(args.output_dir + '/debug'):
            os.makedirs(args.output_dir + '/debug')
             
             
        save_image(preprocessed, 
                   file_dir=args.output_dir + '/debug'  + '/preprocessed.jpg')
    
        contours_img = img.copy()
        
        for i, cand in enumerate(candidates):
            cv2.polylines(contours_img, [cand['approx']], True, (0, 255, 0), thickness=3)
        
        save_image(contours_img, 
                   file_dir=args.output_dir + '/debug'  + '/contours.jpg')
        
        selected_contour_img = img.copy()
        
        cv2.polylines(selected_contour_img, [biggest_cand['approx']], True, (0, 255, 0), thickness=3)
        
        save_image(selected_contour_img, 
                   file_dir=args.output_dir + '/debug'  + '/selected_contour.jpg')
        