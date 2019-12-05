"""
Abdeali
05-12-2019, 13:20
"""

"""This is a package that wil be used by the server to detect faces and recognize the detected faces
Importing all the required modules"""
import cv2
import numpy as np
import os
from keras.models import load_model
""" This reconet is a module of our machine learning model that will give us predictions and 
prepare images to send it for predictions."""
from reconet import Reconet
import keras.backend as K


"""This is the loss function of our machine learning model. We have to import it here because it is 
a custom loss funtion not present in keras library and we have to load it explicitly when loading our
trained model"""
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

"""This class contains all the functions that will be used by server for recognization purpose."""
class Play():
    """Here we initialize the model from reconet file"""
    def __init__(self):
        self.rec = Reconet()
        self.persons = list()
        self.namelabels = list()
        """We prepare one image for every person in our database to check the similarity score.
        The person having the lowest distance between the caputured image is the person in the image."""
        for person in os.listdir("Dataset"):
            self.namelabels.append(person)
            pr = self.rec.prepare_images("Dataset/"+str(person))
            pr = pr[0].reshape((1, 200, 200, 3))
            self.persons.append(pr)
        self.model = load_model("model45.h5", custom_objects = {'contrastive_loss': contrastive_loss})
        """We use haar features to perform face detection and get the coordinates
        of the face in the image"""
        self.faceCascade = cv2.CascadeClassifier("Assets/haarcascade_frontalface_default.xml")
    
    """This function detects faces and returns the coordinates to the server file"""
    def give_me_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30)
        )
        return faces
    """This function performs face recognization using the deteted faces and the faces available in
    the database"""
    def give_me_names(self, frame):
        """We cannot send the image we got from the server directly to the model. So, first
        we prepare the image so that it can be sent to the model."""
        frame = cv2.resize(frame, (200, 200))
        frame = np.array(np.reshape(frame, (200, 200, 3)))/255
        c = frame.reshape((1, 200, 200, 3))
        
        """Once all the images are prepared we send pairs of images to the model to calculate the distance
        between them. in pairs we have one image from the database and one image is of the person
        detected. Then we return the name of the person detected to the server"""
        all_pred = list()
        for per in self.persons:
            pred = self.rec.predict([c, per])
            all_pred.append(pred)
        ind = all_pred.index(min(all_pred))
        return self.namelabels[ind]