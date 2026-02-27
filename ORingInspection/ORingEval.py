import cv2 as cv
import numpy as np
import time


# threshold function method
def threshold(img, t):
    bw = np.zeros_like(img)
    bw[img > t] = 255
    return bw


# binary morphology methods
def dilation(img, k=3):
    pad = k // 2
    padded = np.pad(img, pad)
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.max(padded[i:i + k, j:j + k]) == 255:
                result[i, j] = 255

    return result


def erosion(img, k=3):
    pad = k // 2
    padded = np.pad(img, pad)
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.min(padded[i:i + k, j:j + k]) == 255:
                result[i, j] = 255

    return result


def closing(img, k=3):
    return erosion(dilation(img, k), k)


# main method
for i in range(1, 16):
    # get time at beginning
    start_time = time.time()

    # read in images
    img = cv.imread('Orings/Oring' + str(i) + '.jpg', 0)

    # get mode using histogram (highest peak), calculate mode using: mode - 0.3 * mode (eliminate shadows, only illumination remains)
    pixels = img.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    mode = np.argmax(hist)
    thresh = mode - 0.3 * mode

    # apply threshold
    bw = threshold(img.copy(), thresh)

    # apply binary morphology
    ring = 255 - bw
    ring_closed = closing(ring, k=3)
    bw_clean = 255 - ring_closed

    # time at end
    processing_time = time.time() - start_time

    # add text
    rgb = cv.cvtColor(bw_clean, cv.COLOR_GRAY2RGB)
    cv.putText(rgb, f"Image{i}", (12, 27), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv.putText(rgb, f"Time: {processing_time:.4f}s", (12, 50), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # display images
    cv.imshow('Oring' + str(i), rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()