import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import skvideo.io

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from Image import Image
from trainNetwork import load_weights
from trainNetwork1 import load_weights1
print cv2.__version__

im_fun = Image()
alphabet = [0,1,2,3,4,5,6,7,8,9] 
# cap = cv2.VideoCapture("demo.avi")
# print cap.isOpened()   # True = read video successfully. False - fail to read video.

cap = cv2.VideoCapture('Videos/video-0.avi')
ret, img = cap.read()
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_g = img[:, :, 1]
kernel = np.ones((2, 2))
ret, img_g_bin = cv2.threshold(img_g, 0, 255, cv2.THRESH_OTSU);
img_b = img[:, :, 2]
ret, img_b_bin = cv2.threshold(img_b, 0, 255, cv2.THRESH_OTSU)

# plt.imshow(img_g_bin,'gray')
# plt.figure()
# plt.imshow(img_b_bin,'gray')
# plt.show()

lines = cv2.HoughLinesP(img_g_bin,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 200,maxLineGap = 20)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

lines1 = cv2.HoughLinesP(img_b_bin,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 200,maxLineGap = 20)
for x1,y1,x2,y2 in lines1[0]:
    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)

line_g = lines[0]
line_b = lines1[0]
# plt.imshow(img)
# plt.show()

# test_bin = im_fun.invert(im_fun.image_bin(im_fun.image_gray(img)))
# selected_test, test_numbers = im_fun.select_roi(img.copy(),test_bin)
# im_fun.display_image(selected_test)
# plt.imshow(test,"gray")
# plt.show()

# NAMESTI NEURONSKU MREZU
model = load_weights1('weights1_1');

i=0
while(ret):
    i +=1;
    ret, frame = cap.read()
    if i % 20 != 0 :
        continue
    if( not ret):
        break
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # POKUSAJ DA GLEDAM SAMO CRVENU BOJU
    #img_r = img[:,:,0]
    #img_r = im_fun.dilate(im_fun.erode(img_r))
    #im_fun.display_image(img_r)

    test_bin = im_fun.invert(im_fun.image_bin(im_fun.image_gray(img)))
    #test_bin = im_fun.invert(im_fun.image_bin(img_r))

    # DA LI RADITI sa diletacijom i erozijom? videti kako se prepoznavanje ponasa
    test_bin_e_d = im_fun.erode(im_fun.dilate(test_bin))

    selected_test, test_numbers = im_fun.select_roi(img.copy(),test_bin)

    # SADA TREBA UBACITI PRONADJENE REGIONE U NEURONSKU

    # moram invertovati brojeve da bolje neuronska skonta
    test_numbers = np.array([ im_fun.invert(test_number) for test_number in test_numbers] )

    test_inputs = im_fun.prepare_for_ann(test_numbers)

    # IDE KROZ TEST_IMPUTS I SVAKI BROJ POSEBNO PRIKAZE KAO I REZULTAT MREZE ZA TAJ BROJ
    # j=0;
    # for test_input in test_inputs:
    #     result = model.predict( np.array(test_input.reshape((1, 28, 28, 1))  , np.float32) )
    #     #print ( result )
    #     print ( im_fun.display_result(result, alphabet)  )
    #     im_fun.display_image(test_numbers[j])
    #     j += 1

    result = model.predict_classes(np.array([test_input.reshape(( 28, 28, 1)) for test_input in test_inputs], np.float32))
    print (result)

    # ISCRTAVANJE FREJMA SA OZNACENIM BROJEVIMA
    im_fun.display_image(selected_test)

cap.release()


#   AKO BUDE BILO PROBLEMA SA PREPOZNAVANJEM BROJEVA SAMO STAVI DA JE ZELENI KANAL = 0, POSTO SU SVE FLEKICE ZELENE BOJE
