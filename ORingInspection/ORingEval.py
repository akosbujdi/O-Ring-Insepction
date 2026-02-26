import cv2 as cv
import numpy as np
import time

def threshold(img,t):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x][y] > t:
                img[x][y] = 255
            else:
                img[x][y] = 0
    return img

for i in range(1,16):
    img = cv.imread('Orings/Oring'+str(i)+'.jpg',0)

    # histogram
    pixels = img.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    # get mode (highest peak)
    mode = np.argmax(hist)
    # set thresh to mode - 45 (eliminate shadows, only illumination remains)
    thresh = mode - 45
    bw = threshold(img,thresh)
    rgb = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)
    cv.putText(rgb,f"Image{str(i)}",(12,27), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv.imshow('Oring'+str(i)+'.jpg',rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()

