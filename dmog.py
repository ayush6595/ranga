import numpy as np
import cv2
from collections import deque
import time
import os
from matplotlib import pyplot as plt


pts = deque(maxlen = 1500)
buf2 = deque(maxlen = 3)
time_start = time.time()+3
time_end = time.time()+10
flag1 = True
flag2 = True
font =cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)



def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)


def conv_color(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower_skin = np.array([115,100,100])
  upper_skin = np.array([225,255,255])
  mask = cv2.inRange(hsv, lower_skin, upper_skin)
  res = cv2.bitwise_and(hsv,hsv, mask= mask)
  gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
  return gray



def skin_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,16,55])
    upper_skin = np.array([23,150,180])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    res = cv2.bitwise_and(hsv,hsv, mask= mask)
    blur = cv2.GaussianBlur(res,(75,75),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    return gray





def find_no():
    trainingData = np.load('knn_data.npz')
    train = trainingData['train']
    trainLabels = trainingData['train_labels']
    knn = cv2.KNearest()
    knn.train(train, trainLabels)
    letter = cv2.imread('temp.png')
    letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
    letter = letter.reshape(1,400)
    letter = np.float32(letter)
    ret, result, neighbors, dist = knn.find_nearest(letter, k=5)
    no =result.astype(int)
    return no[0][0]



def find_path(result):
    os.system("chmod +x path.sh")
    os.system("./path.sh " +str(result) )
    return





t_minus = skin_color(cam.read()[1])
t = skin_color(cam.read()[1])
t_plus = skin_color(cam.read()[1])



while(1):
    ret,image = cam.read()
    image_inv = cv2.flip(image,3)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    frame3 = diffImg(t_minus, t, t_plus)
    frame  = cv2.flip(frame3,3)
    ret,thresh1 = cv2.threshold(frame,50,255,cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh1,100,200)
    ret,thresh2 = cv2.threshold(edges,5,255,cv2.THRESH_BINARY)
    #cv2.imshow('thresh2',thresh2)
    hierarchy, contours, _ = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_inv,contours,-1,(0,255,0),2)

    max_x = 0
    max_y = 0
    max_cnt = 0

    for i in range(len(contours)):
        cnt = contours[i]
        if (np.all(max_cnt)<=np.all(cnt)):
         max_cnt = cnt
        top = tuple(max_cnt[max_cnt[:,:,1].argmin()][0])
        x,y =top
        if ((max_x<=x or max_y<=y)):
             extTop = top
    if time.time() >= time_start and time.time() <= time_end:
            if(time.time()-time_start<=4):
                cv2.putText(image_inv,' DRAW ',(30,30),font,1,(0,0,255),8)
            pts.append(extTop)
            cv2.circle(frame,extTop,8,(0,0,0),-1)


    for i in range(1, len(pts)):
            if pts[i - 2] is None or pts[i - 1] is None or pts[i] is None:
                continue
            x1,y1 = extTop
            x2,y2 = pts[len(pts)-1]
            d1 = abs(x2-x1)
            d2 = abs(y2-y1)
            #if d1<160 or d2<160:
            cv2.line(image_inv, pts[i - 1], pts[i], (112, 0, 0),6)
                #cv2.circle(image_inv,pts[i],8,(0,0,0),-1)


    if time.time()>=time_end:

        gray2 = conv_color(image_inv)
        ret3,thresh3 = cv2.threshold(gray2,180,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ret4,thresh4 = cv2.threshold(gray2,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours2, hierarchy2 = cv2.findContours(gray2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_inv,contours2,-1,(0,255,0),2)
        max_area2 =0
        ci2 =0



        for i in range(len(contours2)):
            cnt = contours2[i]
            area = cv2.contourArea(cnt)
            if(area>max_area2):
                max_area2=area
                ci2=i


        cnts = contours2[ci2]
        x,y,w,h = cv2.boundingRect(cnts)
        if x-20>=0:
         x = x-20
        if y-20>=0:
         y = y-20
        if w+20<=640:
         w = w+20
        if h+20<480:
         h = h+20
        roi = thresh4[y:y+h,x:x+w]
        cv2.rectangle(image_inv,(x,y),(x+w+20,y+h+20),(0,0,255),2)
        roi = cv2.resize(roi,(20, 20), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('temp.png',roi)
    cv2.imshow('frame',image_inv)
    t_minus = t
    t = t_plus
    t_plus = skin_color(cam.read()[1])
    if time.time()>=time_end+2:
        break

    k = cv2.waitKey(30) & 0xff
    if k == 27:
     break




cam.release()
cv2.destroyAllWindows()
find_no()
find_path(find_no())
