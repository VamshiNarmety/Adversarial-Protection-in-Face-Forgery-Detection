import os
import cv2
import numpy as np
import dlib
from glob import glob
from imutils import face_utils
from tqdm import tqdm
from multiprocessing import Pool
#define the path of the datasets
CELEB_TRAIN_FAKE = 'venv/data/Celeb DF(v2)/train/Fake'
CELEB_TRAIN_REAL = 'venv/data/Celeb DF(v2)/train/Real'
CELEB_VAL_FAKE = 'venv/data/Celeb DF(v2)/validate/Fake'
CELEB_VAL_REAL = 'venv/data/Celeb DF(v2)/validate/Real'

#Initialize dlib's face detector and a face landmark predictor model
face_detector = dlib.get_frontal_face_detector()
predictor_path = 'venv/src/preprocess/shape_predictor_81_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)

#function to crop faces from the images
def process_facesfromimages(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f'Failed to load image:{image_path}')
        return
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #detect faces
    faces = face_detector(rgb_image, 1)
    if len(faces)==0:
        print(f'No faces detected in {image_path}')
        return
    #choose the largest face and extract landmarks
    face = max(faces, key=lambda rect:rect.width()*rect.height())
    landmarks = face_predictor(rgb_image, face)
    landmarks = face_utils.shape_to_np(landmarks)
    #define bounding box coordinates based on the landmarks
    x0, y0 = landmarks[:, 0].min(), landmarks[:, 1].min() #top-left corner
    x1, y1 = landmarks[:, 0].max(), landmarks[:, 1].max() #bottom-right corner
    #crop the face from the image using bounding box co-ordinates
    cropped_face = image[y0:y1, x0:x1]
    #determine the save path for the cropped face images
    if cropped_face.size>0:
       save_path = image_path.replace('train', 'faces_train').replace('validate', 'faces_validate')
       os.makedirs(os.path.dirname(save_path), exist_ok=True)
       #save the cropped face image to the designated path
       cv2.imwrite(save_path, cropped_face)
    else:
        print(f'Failed to crop face from {image_path}. The cropped image is empty.')
    
#main function
if __name__=="__main__":
    celeb_images = glob(f"{CELEB_TRAIN_FAKE}/*.jpg")+glob(f"{CELEB_TRAIN_REAL}/*.jpg")+glob(f"{CELEB_VAL_FAKE}/*.jpg")+glob(f"{CELEB_VAL_REAL}/*.jpg")
    #use multiprocessing for improving performance by parallel execution
    with Pool(processes=12) as pool:
        pool.map(process_facesfromimages, celeb_images)
        

        
    
        