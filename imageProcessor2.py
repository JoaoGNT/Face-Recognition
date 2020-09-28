"""
Autor: João Guilherme do Nascimento Teles
"""
import glob
import math

import cv2
import dlib

dataBase = 'C:/Users/Windows 10/PycharmProjects/Face-Recognition/DATA/90/*.jpg'
imageTitle = []
labels = []
dataMatrix = []
counter = 0
vecImages = []

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

    if imageTitle[counter][61:74] == 'Nicole_Kidman':
        labels.append(1)
        counter = counter + 1
    else:
        labels.append(-1)
        counter = counter + 1
    labels = labels
    vecCoordinates = []
    vecDetections = []

    for k, d in enumerate(detections):  # For each detected face
        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(0, 67):  # There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0),
                       thickness=-1)  # For each point, draw a red circle with thickness2 on the original frame
            vecDetections.append([shape.part(i).x, shape.part(i).y])

        vecCoordinates.append(vecDetections)

    vecImages.append(vecCoordinates)

euclidianDistVec = []
distVectors = []

for i in range(0, len(vecImages)):
    for j in range(0, 67):
        for a in range(0, 67):
            euclidianDist = math.sqrt(((vecImages[i][0][j][0] - vecImages[i][0][a][0]) ** 2 + (
                    vecImages[i][0][j][1] - vecImages[i][0][a][1]) ** 2))
            euclidianDistVec.append(euclidianDist)

# print(euclidianDistVec)
length = len(euclidianDistVec)
n = 90
for i in range(n):
    start = int(i * length / n)
    end = int((i + 1) * length / n)
    dataMatrix.append(euclidianDistVec[start:end])
