import keras
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from xlwt import Workbook
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from utils import LRN2D
import utils
import sys
import pickle 

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.set_printoptions(threshold=sys.maxsize)

# Loading model as saved from model.py previously
model = keras.models.load_model('my_model.h5', custom_objects={'tf': tf})

# Loading input_embeddings dictionary as saved from model.py previously.
pickle_in = open("input_embeddings.pickle","rb")
input_embeddings = pickle.load(pickle_in)

# *image_to_embedding* function passes an image to the Inception network to generate the embedding vector.
def image_to_embedding(image, model):
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

# About recognize_face function:
# This function calculates the similarity between the to-do inferencing image and the images that are stored in input_embeddings. 
# It passes the image to the trained neural network to generate its embedding vector. 
# Which is then compared with all the embedding vectors of the images in input_images that are stored by calculating L2 Euclidean distance. 
# If the minimum value of L2 distance between two embeddings is less than a threshold then we have a match. 
# Taken the threshold value as 1 (can be adjusted according to the place of inferencing).
def recognize_face(face_image, input_embeddings, model):
    embedding = image_to_embedding(face_image, model)
    minimum_distance = 200
    name = None
    for(input_name, input_embedding) in input_embeddings.items():
        euclidean_distance = np.linalg.norm(embedding-input_embedding)
        print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))
        if euclidean_distance<minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
    if minimum_distance<1:
        return str(name)
    else:
        return None

# This function capture image from the webcam, detect a face in it and crop the image to have a face only, 
# which is then passed to recognize_face function.
def recognize_faces(input_embeddings):
    vc = cv2.VideoCapture('./input/test.mp4')
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while vc.isOpened():
        retval,frame = vc.read()
        frame = cv2.flip(frame,-1)
        try:
            frame.shape
            # print("checked for shape".format(frame.shape))
        except AttributeError:
            # print("shape not found")
            break        
        # if frame.all() == None:
        #     break
        # print("\n")
        # print("\n")
        # print(frame)
        # print("\n")
        # print("\n")
        img = frame
        height, width, channels = frame.shape    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Loop through all the faces detected 
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]    
            identity = recognize_face(face_image, input_embeddings, model)
            if identity is not None:
                if identity not in identities:
                    identities.append(identity)
        
        key = cv2.waitKey(100)
        cv2.imshow("Face Recognition check", img)
        if key == 27: # exit on ESC
            break
    vc.release()
    cv2.destroyAllWindows()
    
identities = []  #will contain list of faces recognized

recognize_faces(input_embeddings)
print("\n")
# printing list
print(identities)

# writing to an excel file record.xls 
wb = Workbook()  
sheet1 = wb.add_sheet('Day 1')
i=0
for item in identities:
    sheet1.write(i,0,identities[i])
    i+=1
wb.save('record.xls')   

