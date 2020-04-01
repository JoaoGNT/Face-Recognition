# Face-Recognition
Evaluation of methodologies for facial recognition

---> This work was developed as a Scientific Project in Federal University of Uberlândia

                                                         Abstract
Several areas of research focused on image analysis were created from the
development of the digitization process. The abundance of data to be explored, derived
from this process, made possible analyses in many different ways. Among these analyzes,
facial recognition is one of the areas of research with several security applications. As a
result, systems that perform this type of analysis have several steps, as applications
require a high degree of reliability. These steps are divided into: face detection, image
suitability to algorithm, description and classification. This paper analyzes classification
methodologies for facial recognition, seeking to use face data as usefully as possible. Two
data classification methodologies were evaluated, the artificial neural networks and the
support vector machine, using as a descriptor the gray level of the image pixels. From
this, tests were performed using 3 databases with 30, 60 and 90 images. In addition, the
following parameters of the classifiers were varied: network architecture for the artificial
neural network and kernel type and the constant C for the support vector machine. After
the tests, the highest hit rate for RNA was 96.67% and for SVM was 93.33%.

                                                    Technical Aspects

This work was built using a JetBrains platform called PyCharm 2019.2.1.
You will need to install the following libraries: OpenCV, Dlib, Numpy, Glob, Scipy and Sklearn.
This code was used to test some types of algorithms for facial recognition, as well as, the performance variation in data sets with different sizes. Two types of algorithms were tested: Support Vector Machine and Artificial Neural Networks along with three different data sets, with: 30, 60 and 90 photos.

Six Python scripts were created and named by the following model: 

•	redeneuralautoXX: corresponds to ANN, and the index XX corresponds to the data set size (30, 60 or 90);

•	svmautoXX: corresponds to SVM, and the index XX corresponds to the data set size (30, 60 or 90).

Each one of the scripts prints two values that corresponds to Hits and Misses of prediction.
                                                  
---> Using the data set that corresponds to the code is important, since the labels used in the code were encoded for a given data folder, such as the name "DADOS", which appears in this same repository.

