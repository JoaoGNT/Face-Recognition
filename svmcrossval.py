"""
Autor: João Guilherme do Nascimento Teles
"""
import cv2
import dlib
import numpy as np
import glob
import scipy.ndimage
from sklearn import svm
from sklearn.model_selection import cross_val_score


class svm30:


    ## -- Files containing the data set --
    arquivo30 = 'C:/Users/Windows 10/PycharmProjects/Face-Recognition/DADOS/30/kf/*.jpg'
    " ^ This field must be changed by the path to the folder containing the Training data set"

    vec2 = []
    vec3 = []
    matrix = []

    detector = dlib.get_frontal_face_detector()  # Detector facial

    # Marcadores
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # /----- IMAGEM A SER IDENTIFICADA ----/


    for i in glob.glob(arquivo30):
        vec = i
        vec2.append(vec)
        tam = np.size(vec2)
        tamanho = str(tam) + ".jpg"
        img = cv2.imread(i)

        frame = cv2.flip(img, 180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1)

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

        matrix.append(v[0])

    matrix = np.array(matrix, np.float32)

    # print(matrix)

    labels = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,-1],np.float32)
    clf = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

    scores = cross_val_score(clf, matrix, labels, cv=5, scoring='accuracy')
    # print(scores)

    Accuracy = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
class svm60:


    ## -- Files containing the data set --
    arquivo60 = 'C:/Users/Windows 10/PycharmProjects/Face-Recognition/DADOS/60/kf/*.jpg'
    " ^ This field must be changed by the path to the folder containing the Training data set"

    vec2 = []
    vec3 = []
    matrix = []

    detector = dlib.get_frontal_face_detector()  # Detector facial

    # Marcadores
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # /----- IMAGEM A SER IDENTIFICADA ----/


    for i in glob.glob(arquivo60):
        vec = i
        vec2.append(vec)
        tam = np.size(vec2)
        tamanho = str(tam) + ".jpg"
        img = cv2.imread(i)

        frame = cv2.flip(img, 180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1)

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

        matrix.append(v[0])

    matrix = np.array(matrix, np.float32)

    # print(matrix)

    labels = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
                       1,1,1,1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      np.float32)
    clf = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

    scores = cross_val_score(clf, matrix, labels, cv=5, scoring='accuracy')
    # print(scores)

    Accuracy = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
class svm90:

    ## -- Files containing the data set --
    arquivo90 = 'C:/Users/Windows 10/PycharmProjects/Face-Recognition/DADOS/90/kf/*.jpg'
    " ^ This field must be changed by the path to the folder containing the Training data set"

    vec2 = []
    vec3 = []
    matrix = []

    detector = dlib.get_frontal_face_detector()  # Detector facial

    # Marcadores
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # /----- IMAGEM A SER IDENTIFICADA ----/


    for i in glob.glob(arquivo90):
        vec = i
        vec2.append(vec)
        tam = np.size(vec2)
        tamanho = str(tam) + ".jpg"
        img = cv2.imread(i)

        frame = cv2.flip(img, 180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1)

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

        matrix.append(v[0])

    matrix = np.array(matrix, np.float32)

    # print(matrix)

    labels = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, -1, -1, -1, -1, -1, -1],
                      np.float32)
    clf = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

    scores = cross_val_score(clf, matrix, labels, cv=5, scoring='accuracy')
    # print(scores)

    Accuracy = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

print(svm30.Accuracy)
print(svm60.Accuracy)
print(svm90.Accuracy)