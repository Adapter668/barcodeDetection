import numpy as np
import cv2
from matplotlib import  pyplot as plt

cap = cv2.VideoCapture(0)
while True:
    ret_val, img4 = cap.read()
    img3 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    sobx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    soby = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)

    gradient = cv2.subtract(sobx, soby)
    gradient = cv2.convertScaleAbs(gradient)

    blur = cv2.blur(gradient, (9, 9))
    ret, thresh = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    img2, cnt, hier = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt) > 0:
        print("znalezione")
        c = sorted(cnt, key=cv2.contourArea, reverse=True)[0]

        rect = cv2.minAreaRect(c)
        print(rect)
        box = np.int0(cv2.boxPoints(rect))

        cv2.drawContours(img4, [box], -1, (0, 255, 0), 3)

    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    cv2.imshow('barcode', img4)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    #plt.imshow(img3)
    #plt.show()