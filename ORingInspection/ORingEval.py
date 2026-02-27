import cv2 as cv
import numpy as np
import time
from collections import deque


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


# connected component labelling method
def connected_components(binary_img):
    h, w = binary_img.shape
    labels = np.zeros((h, w), dtype=int)
    label = 0
    areas = {}

    for i in range(h):
        for j in range(w):

            if binary_img[i, j] == 255 and labels[i, j] == 0:

                label += 1
                area = 0
                q = deque()
                q.append((i, j))
                labels[i, j] = label

                while q:
                    x, y = q.popleft()
                    area += 1

                    # 4-connectivity
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < h and 0 <= ny < w:
                            if binary_img[nx, ny] == 255 and labels[nx, ny] == 0:
                                labels[nx, ny] = label
                                q.append((nx, ny))

                areas[label] = area

    return labels, areas


# main section
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

    # apply connected component labelling
    labels, areas = connected_components(ring_closed)
    # print(f"Image {i} -> Components found: {len(areas)}") # return number of components found in the image

    if len(areas) > 0:
        largest_label = max(areas, key=areas.get)

        # Create image containing only largest component
        ring_only = np.zeros_like(ring_closed)
        ring_only[labels == largest_label] = 255
    else:
        ring_only = ring_closed.copy()

    # Convert back to black ring / white background
    bw_final = 255 - ring_only

    # time at end
    processing_time = time.time() - start_time

    # add text
    rgb = cv.cvtColor(bw_final, cv.COLOR_GRAY2RGB)
    cv.putText(rgb, f"Image{i}", (12, 27), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv.putText(rgb, f"Time: {processing_time:.4f}s", (12, 50), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # display images
    cv.imshow('Oring' + str(i), rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()
