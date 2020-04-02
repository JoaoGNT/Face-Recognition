"""
Autor: João Guilherme do Nascimento Teles
"""
import cv2
import dlib
import numpy as np
import glob
import scipy.ndimage
from sklearn import svm


## -- Files containing the data set --
arquivo = 'C:/Users/Windows 10/PycharmProjects/Face-Recognition/DADOS/30/Training/*.jpg'
" ^ This field must be changed by the path to the folder containing the Training data set"
arquivo1 = 'C:/Users/Windows 10/PycharmProjects/Face-Recognition/DADOS/30/Predict/*.jpg'
" ^ This field must be changed by the path to the folder containing the Prediction data set"
## --

vec2 = []
vec3 = []
matriztreinamento = []

detector = dlib.get_frontal_face_detector()  # Detector facial

# Marcadores
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#/----- IMAGEM A SER IDENTIFICADA ----/


for i in glob.glob(arquivo):
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
    #cv2.imwrite(tamanho, clahe_image)
    #cv2.imwrite("a" + tamanho, zoom)

    v = np.resize(zoom, (1, 100 * 100))
    v = np.array(v, np.float32)
    #print(v)

    matriztreinamento.append(v[0])


matriztreinamento = np.array(matriztreinamento, np.float32)

print(matriztreinamento)

# -------- SVM --------#

#[0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0]
labels = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1], np.float32)
clf = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(matriztreinamento,labels)


kk=1
acertos = 0
erros = 0


for j in glob.glob(arquivo1):
    vec = j
    #vec2.append(vec)
    #tam = np.size(vec2)
    #tamanho = str(tam)+".jpg"
    imagem_carregada = cv2.imread(j)

    frame1 = cv2.flip(imagem_carregada, 180)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    clahe_image1 = clahe.apply(gray1)
    detections1 = detector(clahe_image1, 1)

    for k, d in enumerate(detections1):  # For each detected face
        shape = predictor(clahe_image1, d)  # Get coordinates
        for i in range(1, 68):  # There are 68 landmark points on each face
            cv2.circle(frame1, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0),
                thickness=-1)  # For each point, draw a red circle with thickness2 on the original frame
    vec1x = []
    vec1y = []

    for i in range(1, 68):
        vec1x.append(shape.part(i).x)
        vec1y.append(shape.part(i).y)

    min1x = np.amin(vec1x)
    min1y = np.amin(vec1y)
    max1x = np.amax(vec1x)
    max1y = np.amax(vec1y)

    clahe_image1 = clahe_image1[min1y:max1y, min1x:max1x]



    zoom1 = scipy.ndimage.zoom(clahe_image1,((100)/(max1y-min1y),(100)/(max1x-min1x)), order = 0)
    v1 = np.resize(zoom1, (1, 100*100))
    v1 = np.array(v1,np.float32)


#/----FIM ----/

    labels2 = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1], np.float32)
    results = clf.predict(np.array(v1, np.float32))
    #print(j,results)
    print(results)


    #if np.sign(results[1][0][0]) == np.sign(labels2[kk-1]):
    if np.sign(results[0]) == np.sign(labels2[kk-1]):
        acertos = acertos+1
    else:
        erros = erros+1

    kk=kk+1

print("Hits:",acertos)
print("Misses:",erros)

# ------ REDE FIM -------#

