import numpy as np
import cv2
from tkinter import *
from matplotlib import  pyplot as plt

class Application(Frame):
    def detection(self, img4):
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
            c = sorted(cnt, key=cv2.contourArea, reverse=True)[0]

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))

            cv2.drawContours(img4, [box], -1, (0, 255, 0), 3)

    def cameraView(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret_val, img4 = cap.read()
            self.detection(img4)
            cv2.imshow('barcode', img4)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def photoView(self, name):
        img = cv2.imread(name)
        height, width, _ = img.shape
        print(height, width)
        h = np.floor(height / 500)
        w = np.floor(width / 500)

        if h > 1 and w > 1:
            x = np.maximum(h, w)
            img4 = cv2.resize(img, (int(width / x), int(height / x)))
        elif h > 1 and w <= 1:
            img4 = cv2.resize(img, (int(width / h), int(height / h)))
        elif w > 1 and h <= 1:
            img4 = cv2.resize(img, (int(width / w), int(height / w)))
        else:
            img4 = cv2.resize(img, (width, height))
        self.detection(img4)
        cv2.imshow("image", img4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def createWidgets(self):
        l = Label(self, text="Path to photo:")
        l.grid(row=0,column=0)
        e = Entry(self)
        e.insert(0, "images/01.jpg")
        e.grid(row=0, column=1)
        photo = Button(self, text="Find barcode on photo", command=lambda: self.photoView(e.get()))
        photo.grid(row=2, columnspan=2)
        camera = Button(self, text="Find barcode on camera view", command=self.cameraView)
        camera.grid(row=3, columnspan=2)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()
