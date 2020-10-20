import cv2
import numpy as np
import os,sys
from preprocessing import *
import tkinter as tk

def crop_text_image(image_path,model):
    

    #import image
    image = cv2.imread(image_path)
    #image = image_path

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    

    #dilation
    kernel = np.ones((5,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)


    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    text = ""
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y+h, x:x+w]
        text = text + crop_text_image_chars(roi,model=model)
        
        # show ROI
        cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        cv2.waitKey(0)
        text = text + '\n'
        


    root = tk.Tk()
    logo = tk.PhotoImage(file="0.png")

    w1 = tk.Label(root, image=logo).pack(side="left")

    explanation = text

    w2 = tk.Label(root, 
              justify=tk.LEFT,
              padx = 20, 
              text=explanation,
              font = "Verdana 16 bold").pack(side="left")
    root.mainloop()



def crop_text_image_chars(image,model):
    
    numbers_array = ""

    image = cv2.resize(image,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)


    #dilation
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)


    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    old_x = 0
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if(i != 0):
            dist = x-old_x
            if(dist > 105):
                numbers_array = numbers_array + " "
        # Getting ROI
        roi = image[y:y+h, x:x+w]
        # show ROI
        roi = resize_image(roi)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = preprocess(roi)

        old_x = x + w
        #predict
        img = roi
        img = img.reshape(1,28,28,1)
        predictions = model.predict(img)
        digit = np.argmax(predictions[0])
         
        digit = int(digit)
        if(digit != 10):
            numbers_array = numbers_array + str(digit)
        ##

        roi = np.fliplr(np.rot90(roi,3))
        cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        cv2.waitKey(0)

    print(numbers_array)
    return numbers_array
    
    #cv2.imshow('marked areas',image)
    #cv2.waitKey(0)

def resize_image(img, size=(28,28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    if dif > (size[0]+size[1])//2:
        interpolation = cv2.INTER_AREA
    else:
        interpolation=cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.full((dif, dif),255, dtype=img.dtype)
        
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.full((dif, dif, c),255, dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)