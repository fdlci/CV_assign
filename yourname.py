import numpy as np 
#from skimage import transform
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from skimage.io import imread
import os
import pickle
import cv2
#from config import *
from skimage import color
import matplotlib.pyplot as plt 

def saving(filename, file_values):
    return pickle.dump(file_values, open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, "rb"))

def sliding_window(im, window_size, step_size):
    for y in range(0, im.shape[0], step_size[1]):
        for x in range(0, im.shape[1], step_size[0]):
            yield (x, y, im[y: y + window_size[1], x: x + window_size[0]])

def pedestrians(path):
    ''' Return a list of bounding boxes in the format [frame, bb_id, x,y,dx,dy]
    '''
    window_size = [64, 128]
    step_size = [8,8]
    downscale = 1.8
    detections = []

    svm_model = load('VIC_Assignment2/svm_model.p')


    for frame_id, image_path in enumerate(os.listdir(path)):
        image = imread(os.path.join(path,image_path),as_gray=True)

        detect_frame = []
        scale = 0
        bb_id = 1
        for im_scaled in pyramid_gaussian(image, downscale = downscale):
            n, p = im_scaled.shape
            if n < window_size[1] or p < window_size[0]:
                break
            
            for (x,y,im_window) in sliding_window(im_scaled, window_size, step_size):
                if im_window.shape[0] != window_size[1] or im_window.shape[1] != window_size[0]:
                    continue
                feature = hog(im_window, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)[0]
                feature = feature.reshape(1,-1)
                prediction = svm_model.predict(feature)

                if prediction == 1:
                    if svm_model.decision_function(feature) > 0.7:
                        rescale = downscale**scale
                        detect_frame.append([frame_id+1, bb_id, int(x*rescale),int(y*rescale), int(window_size[0]*rescale), int(window_size[1]*rescale)])
                        detections.append([frame_id+1, bb_id, int(x*rescale),int(y*rescale), int(window_size[0]*rescale), int(window_size[1]*rescale)])
                        bb_id += 1

            scale += 1
        
        saving('detect_'+str(frame_id+1)+'.p', detect_frame)
    
    
    return detections

if __name__ == '__main__':

    path = 'img1'
    # path = 'Images_to_test'
    detections = pedestrians(path)

