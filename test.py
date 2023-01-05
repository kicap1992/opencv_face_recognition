import cv2
import os
from flask import Flask,request,render_template, redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

faces = []
labels = []
userlist = os.listdir('static/faces')
for user in userlist:
    for imgname in os.listdir(f'static/faces/{user}'):
        img = cv2.imread(f'static/faces/{user}/{imgname}')
        resized_face = cv2.resize(img, (50, 50))
        faces.append(resized_face.ravel())
        labels.append(user)
faces = np.array(faces)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces,labels)
joblib.dump(knn,'static/face_recognition_model.pkl')