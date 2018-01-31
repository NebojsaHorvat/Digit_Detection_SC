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

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

im_fun = Image()
alphabet = [0,1,2,3,4,5,6,7,8,9]

f = open('out.txt', 'w')
f.write('RA 77/2014 Nebojsa Horvat')
f.write('file	sum')
videos = ['video-0.avi','video-1.avi','video-2.avi','video-3.avi','video-4.avi','video-5.avi','video-6.avi','video-7.avi','video-8.avi','video-9.avi',]

for video in videos:
    SUM = 0;
    # cap = cv2.VideoCapture("demo.avi")
    # print cap.isOpened()   # True = read video successfully. False - fail to read video.
    redni_br = 0
    ID_variable = 0
    cap = cv2.VideoCapture('Videos/%s'%(video))
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
    line_g = line_g[0]
    line_b = lines1[0]
    line_b = line_b[0]
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
    # found_numbers_everything      ovde cu suvati LISTU  [ ( ID, sliku,   [ (u kolko iteracija se nije pojavila, (X,Y) koordinatu poslednje pojave, predikciju )]    )]
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
        #print (result)

        result_iter=0
        for img_with_cord in number_imgs_with_coord:
            img_with_cord.append(result[result_iter])
            result_iter += 1

        # AKO JE PRVI FRAME SAMO UBACIM SVE STO SAM NASAO U NJEMU
        if( i == 1 ):
             # u found_numbers_everything cu  suvati LISTU  [  ID, sliku, u kolko iteracija se nije pojavila [ ( X koordinatu poslednje pojave,Y koordinatu poslednje pojave, predikciju )], zelena, plava  ]
            j = 0;
            for img_with_cord in number_imgs_with_coord:
                found_numbers_everything.append( [ID_variable, im_fun.invert(number_imgs[j]) ,0, [( number_imgs_with_coord[j][1][0],number_imgs_with_coord[j][1][1],result[j] )],False,False  ]  )
                ID_variable +=1;
                j += 1;

        else:
            j=0;
            number_imgs_with_coord_without_found = number_imgs_with_coord
            for fount_number_everything in found_numbers_everything:
                w, h = fount_number_everything[1].shape[::-1]
                # print '%d'% (fount_number_everything[0])
                # print fount_number_everything[2][0][2]
                # print '\n'
                res = cv2.matchTemplate(frame_bin_before_erode, fount_number_everything[1] ,cv2.TM_CCOEFF_NORMED )
                threshold = 0.58  # sto je veci broj to je slabiji uslov (0.65 je radio posao dobro valjda)
                loc = np.where( res >= threshold)
                found = False;
                for pt in zip(*loc[::-1]):
                    if(     ( pt[0] <= fount_number_everything[3][-1][0] - 5 + (fount_number_everything[2] * 2*12 ) )
                    or ( pt[1] <= fount_number_everything[3][-1][1] - 5 + (fount_number_everything[2] * 2*12) )
                    or  ( pt[0] > fount_number_everything[3][-1][0] + 30 + (fount_number_everything[2] * 2*20 )  )
                    or ( pt[1] > fount_number_everything[3][-1][1] + 30 + (fount_number_everything[2] *2*20 ) )       ):
                        continue;
                    last_prediction = fount_number_everything[3][-1][2];
                    fount_number_everything[3].append( (pt[0],pt[1],last_prediction) )
                    cv2.rectangle(selected_test, pt, (pt[0] + w, pt[1] + h), im_fun.get_color( fount_number_everything[0] )*255, 2)
                    found = True;

                    # SADA TREBA DA PRODJEM KROZ SVE KONTURE KOJE SAM NASAO U OVOM FREJMU I DA IZBACIM ONU KOJA JE VEC PRONADJENA
                    x1 = pt[0] # gornja x kordinata trenutno pronadjenog broja
                    y1 = pt[1] # gornja y kordinata trenutno pronadjenog broja
                    x2 = pt[0] + 15; # donja x kordinata trenutno pronadjenog broja
                    y2 = pt[1] + 15; # donja y kordinata trenutno pronadjenog broja
                    number_imgs_with_coord_without_found_empy=[]
                    for img_with_cord in number_imgs_with_coord_without_found:
                        x = img_with_cord[1][0]
                        y = img_with_cord[1][1]
                        if not ( x2 > x > x1 and x2 > x > x1):
                            number_imgs_with_coord_without_found_empy.append(img_with_cord)
                    number_imgs_with_coord_without_found = number_imgs_with_coord_without_found_empy
                    break;
                if found:
                    fount_number_everything[2] = 0;
                    # ovde cu gledati da li je broj bio nestao i ako jeste kuda je onda prosao

                    # PROVERAVAM ZA ZELENU
                    if( not fount_number_everything[4]):
                        A_n = [fount_number_everything[3][-1][0] , fount_number_everything[3][-1][1] ]
                        B_n = [fount_number_everything[3][-2][0] , fount_number_everything[3][-2][1] ]
                        C_l = [line_g[0] , line_g[1]]
                        D_l = [line_g[2] , line_g[3]]

                        if( intersect(A_n,B_n,C_l,D_l)):
                            fount_number_everything[4] = True;
                            SUM -= fount_number_everything[3][-1][2]
                            #print 'Oduzeo sam %d   SUMA :%d'%(fount_number_everything[3][-1][2],SUM)
                    # PROVERAVAM ZA PLAVU
                    if( not fount_number_everything[5]):
                        A_n = [fount_number_everything[3][-1][0] , fount_number_everything[3][-1][1] ]
                        B_n = [fount_number_everything[3][-2][0] , fount_number_everything[3][-2][1] ]
                        C_l = [line_b[0] , line_b[1]]
                        D_l = [line_b[2] , line_b[3]]

                        if( intersect(A_n,B_n,C_l,D_l)):
                            fount_number_everything[5] = True;
                            SUM += fount_number_everything[3][-1][2]
                            #print 'Sabrao sam %d   SUMA :%d'%(fount_number_everything[3][-1][2],SUM)
                else:
                    fount_number_everything[2] += 1;
                    # print '%d not fount iterations %d NUMBER: %d'% (fount_number_everything[0] ,fount_number_everything[2],fount_number_everything[3][-1][2])
                j +=1
            # SADA ONE KOJE NISAM IZBACIO TREBA DA UBACIM U LISTU KAO NOVE
            k = 0;
            for  number_imgs_with_coord in number_imgs_with_coord_without_found:
                found_numbers_everything.append( [ID_variable, number_imgs_with_coord[0] ,0, [( number_imgs_with_coord[1][0],number_imgs_with_coord[1][1],number_imgs_with_coord[2] )],False,False  ]  )
                ID_variable +=1;
                k += 1;
            #print 'Dodato novih'
            #print [( number_imgs_with_coord[2],(number_imgs_with_coord[1][0],number_imgs_with_coord[1][1])) for  number_imgs_with_coord in number_imgs_with_coord_without_found]
            #print '\n'

        # ISCRTAVANJE FREJMA SA OZNACENIM BROJEVIMA
        #im_fun.display_image(selected_test)
    f.write('%s\t%d\n'% (video,SUM))
    print '%s SUMA : %d'% (video,SUM)
    cap.release()

f.close()

#   AKO BUDE BILO PROBLEMA SA PREPOZNAVANJEM BROJEVA SAMO STAVI DA JE ZELENI KANAL = 0, POSTO SU SVE FLEKICE ZELENE BOJE
