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
redni_br = 0
cap = cv2.VideoCapture('Videos/video-8.avi') # u 3 ne readi pred kraj
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

# PROMENJLIVE ZA CUVANJE REZULTATA
# found_numbers_everything      ovde cu suvati LISTU  [ ( sliku,   [ (u kom frejsmu se pojavi, (X,Y) koordinatu, predikciju )]    )]
found_numbers_everything = [];
i=0

# za testiranje promenljive
first_img = 0;
last_fount = [[0,0],[0,0],[0,0]]
not_foun_iterations = [0, 0 ,0 ]

while(ret):
    ret, frame = cap.read()
    i += 1
    if i % 20 != 1 :
        continue
    if( not ret):
        break
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # POKUSAJ DA GLEDAM SAMO CRVENU BOJU
    #img_r = img[:,:,0]
    #img_r = im_fun.dilate(im_fun.erode(img_r))
    #im_fun.display_image(img_r)

    frame_bin = im_fun.invert(im_fun.image_bin(im_fun.image_gray(img)))
    #test_bin = im_fun.invert(im_fun.image_bin(img_r))

    # DA LI RADITI sa diletacijom i erozijom? videti kako se prepoznavanje ponasa
    # test_bin_e_d = im_fun.erode(im_fun.dilate(test_bin))

    # uradi eroziju binarne slike frejma da vidis hoce li bolje prepoznati
    frame_bin_before_erode = frame_bin

    # plt.imshow(frame_bin,'gray')
    # plt.figure()
    frame_bin = im_fun.erode(frame_bin);
    selected_test, number_imgs, number_imgs_with_coord = im_fun.select_roi(img.copy(),frame_bin)
    # plt.imshow(frame_bin,'gray')
    # plt.show();


    # SADA TREBA UBACITI PRONADJENE REGIONE U NEURONSKU

    # moram invertovati brojeve da bolje neuronska skonta
    number_imgs = np.array([ im_fun.invert(test_number) for test_number in  number_imgs] )

    test_inputs = im_fun.prepare_for_ann(number_imgs)

    # IDE KROZ TEST_IMPUTS I SVAKI BROJ POSEBNO PRIKAZE KAO I REZULTAT MREZE ZA TAJ BROJ
    # j=0;
    # for test_input in test_inputs:
    #     result = model.predict( np.array(test_input.reshape((1, 28, 28, 1))  , np.float32) )
    #     #print ( result )
    #     print ( im_fun.display_result(result, alphabet)  )
    #     im_fun.display_image(number_imgs[j])
    #     j += 1

    result = model.predict_classes(np.array([test_input.reshape(( 28, 28, 1)) for test_input in test_inputs], np.float32))
    print (result)



    # AKO JE PRVI FRAME SAMO UBACIM SVE STO SAM NASAO U NJEMU
    if( i == 1 ):
        # found_numbers_everything      ovde cu suvati LISTU  [ ( sliku,   [ (u kom frejsmu se pojavi, (X,Y,W,H) koordinatu, predikciju )]    )]
        # j = 0;
        # for img_with_cord in number_imgs_with_coord
        #     found_numbers_everything.append( (  img_with_cord[0]  ,  [(  )]  ))
        #     j += 1;

        first_imgs =[ im_fun.invert(number_imgs[redni_br]) ]
        last_fount[0][0] = number_imgs_with_coord[redni_br][1][0]
        last_fount[0][1] = number_imgs_with_coord[redni_br][1][1]
        first_imgs.append( im_fun.invert(number_imgs[redni_br + 1 ]) )
        last_fount[1][0] = number_imgs_with_coord[redni_br + 1][1][0]
        last_fount[1][1] = number_imgs_with_coord[redni_br + 1][1][1]
        first_imgs.append( im_fun.invert(number_imgs[redni_br + 2 ]) )
        last_fount[2][0] = number_imgs_with_coord[redni_br + 2][1][0]
        last_fount[2][1] = number_imgs_with_coord[redni_br + 2][1][1]
    else:
        j=0;

        for first_img in first_imgs:
            w, h = first_img.shape[::-1]

            res = cv2.matchTemplate(frame_bin_before_erode,first_img,cv2.TM_CCOEFF_NORMED )
            threshold = 0.65  # sto je veci broj to je slabiji uslov (0.65 je radio posao dobro valjda)
            loc = np.where( res >= threshold)

            found = False;
            for pt in zip(*loc[::-1]):
                if(    ( ( pt[0] <= last_fount[j][0] - 5 ) or ( pt[1] <= last_fount[j][1] - 5) )
                or(  ( pt[0] > last_fount[j][0] + 30 + (not_foun_iterations[j] * 2*20)  ) or ( pt[1] > last_fount[j][1] + 30 + (not_foun_iterations[j] *2*20 ) ) )      ):
                    # print 'xProsli:  %d, xTrenutni: %d' % ( last_fount[j][0], pt[0] )
                    # print 'yProsli:  %d, xTrenutni: %d' % ( last_fount[j][1], pt[1])
                    # print 'not fount iterations %d'% (not_foun_iterations[j])
                    # print j
                    continue;
                last_fount[j][0] = pt[0]
                last_fount[j][1] = pt[1]

                if(j == 0):
                    color = (255,0,0)
                elif (j == 1):
                    color = (0,0,255)
                else:
                    color = (255,255,0)
                cv2.rectangle(selected_test, pt, (pt[0] + w, pt[1] + h), color, 2)
                found = True;
                break;
            if found:
                not_foun_iterations[j] = 0;
            else:
                not_foun_iterations[j] += 1;
                print '%d not fount iterations %d'% (j ,not_foun_iterations[j])
            j += 1;
    # ISCRTAVANJE FREJMA SA OZNACENIM BROJEVIMA
    im_fun.display_image(selected_test)



cap.release()


#   AKO BUDE BILO PROBLEMA SA PREPOZNAVANJEM BROJEVA SAMO STAVI DA JE ZELENI KANAL = 0, POSTO SU SVE FLEKICE ZELENE BOJE
