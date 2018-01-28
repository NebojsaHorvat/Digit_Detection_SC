import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import skvideo.io
print cv2.__version__

# cap = cv2.VideoCapture("demo.avi")
# print cap.isOpened()   # True = read video successfully. False - fail to read video.

cap = cv2.VideoCapture('video-1.avi')
ret, img = cap.read()
print img.shape
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
plt.imshow(img)
plt.show()


while(ret):
    #todo
    ret, frame = cap.read()
    #plt.imshow(frame);
    #plt.show()
cap.release()
