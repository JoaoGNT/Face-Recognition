"""
Autor: João Guilherme do Nascimento Teles
"""
import glob

import cv2
import dlib
import numpy as np
import scipy.ndimage

dataBase = 'C:/Users/Windows 10/PycharmProjects/Face-Recognition/DATA/90/*.jpg'
imageTitle = []
labels = []
dataMatrix = []
counter = 0

detector = dlib.get_frontal_face_detector()  # Detector facial
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

for image in glob.glob(dataBase):
    img = cv2.imread(image)
    imageTitle.append(image.title())

    frame = cv2.flip(img, 180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)

    if imageTitle[counter][62:75] == 'Nicole_Kidman':
        labels.append(1)
        counter = counter + 1
    else:
        labels.append(-1)
        counter = counter + 1
    labels = labels
    for k, d in enumerate(detections):  # For each detected face
        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(1, 68):  # There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0),
                       thickness=-1)  # For each point, draw a red circle with thickness2 on the original frame

    vecx = []
    vecy = []
    for i in range(1, 68):
        vecx.append(shape.part(i).x)
        vecy.append(shape.part(i).y)

    minx = np.amin(vecx)
    miny = np.amin(vecy)
    maxx = np.amax(vecx)
    maxy = np.amax(vecy)

    clahe_image = clahe_image[miny:maxy, minx:maxx]

    zoom = scipy.ndimage.zoom(clahe_image, ((100) / (maxy - miny), (100) / (maxx - minx)), order=0)

    v = np.resize(zoom, (1, 100 * 100))
    v = np.array(v, np.float32)

    dataMatrix.append(v[0])

dataMatrix = np.array(dataMatrix, np.float32)
