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


def classify_ring(ring_only, raw_ring, img_index):
    # get centroid
    coords = np.argwhere(ring_only == 255)
    if len(coords) == 0:
        return "FAIL", (0, 0, 255)

    cy = int(np.mean(coords[:, 0]))
    cx = int(np.mean(coords[:, 1]))

    raw_coords = np.argwhere(raw_ring == 255)
    if len(raw_coords) == 0:
        return "FAIL", (0, 0, 255)

    # calculate angle and distance of every ring pixel from centroid
    raw_angles = np.degrees(np.arctan2(raw_coords[:, 0] - cy, raw_coords[:, 1] - cx))
    raw_angles_int = np.round(raw_angles).astype(int)  # bucketed to nearest degree
    distances = np.sqrt((raw_coords[:, 0] - cy) ** 2 + (raw_coords[:, 1] - cx) ** 2)

    # build outer edge profile: for each angle keep only the furthest pixel
    outer_radius = {}
    for a, d in zip(raw_angles_int, distances):
        if a not in outer_radius or d > outer_radius[a]:
            outer_radius[a] = d

    # compare the minimum outer radius to the mean across all angles
    outer_radii = np.array(list(outer_radius.values()))
    outer_dip = np.min(outer_radii) / np.mean(outer_radii)
    # print(f"Image {img_index} -> Outer dip ratio: {outer_dip:.4f}") # debugging

    # perfect ring near 1.0, values below classified as FAIL
    CHIP_THRESHOLD = 0.95

    if outer_dip < CHIP_THRESHOLD:
        return "FAIL", (0, 0, 255), outer_dip
    else:
        return "PASS", (0, 200, 0), outer_dip


# main section
for i in range(1, 16):
    # get time at beginning
    start_time = time.time()

    # read-in images
    img = cv.imread('Orings/Oring' + str(i) + '.jpg', 0)

    # get mode using histogram (highest peak), calculate mode using: mode - 0.3 * mode (eliminate shadows, only illumination remains)
    pixels = img.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    mode = np.argmax(hist)
    thresh = mode - 0.3 * mode

    # --- apply threshold ---
    bw = threshold(img.copy(), thresh)

    # apply binary morphology
    ring = 255 - bw
    ring_closed = closing(ring, k=3)

    # --- apply connected component labelling ---
    labels, areas = connected_components(ring_closed)
    # print(f"Image {i} -> Components found: {len(areas)}") # return number of components found in the image (debugging)

    if len(areas) > 0:
        largest_label = max(areas, key=areas.get)

        # Create image containing only largest component
        ring_only = np.zeros_like(ring_closed)
        ring_only[labels == largest_label] = 255
    else:
        ring_only = ring_closed.copy()

    # --- apply classification (PASS/FAIL) ---
    label_text, label_colour, dip_score = classify_ring(ring_only, ring, i)

    # Convert back to black ring / white background
    bw_final = 255 - ring_only

    # time at end
    processing_time = time.time() - start_time

    # --- visualization ---
    rgb = cv.cvtColor(bw_final, cv.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]

    # top left - image index
    cv.putText(rgb, f"Image {i}:", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 150, 0), 2)

    # bottom right - score/time
    time_size, _ = cv.getTextSize(f"Time: {processing_time:.4f}s", cv.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    score_size, _ = cv.getTextSize(f"Score: {dip_score:.4f}", cv.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv.putText(rgb, f"Time: {processing_time:.4f}s", (w - time_size[0] - 5, h - score_size[1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.52, (255, 150, 0), 2)
    cv.putText(rgb, f"Score: {dip_score:.4f}", (w - score_size[0] - 5, h - 5), cv.FONT_HERSHEY_SIMPLEX, 0.52,
               (255, 150, 0), 2)

    # centre - classification result
    text_size, _ = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    cv.putText(rgb, label_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1.1, label_colour, 3)

    # display images + add border
    border_colour = label_colour
    rgb = cv.copyMakeBorder(rgb, 6, 6, 6, 6, cv.BORDER_CONSTANT, value=border_colour)
    cv.imshow('Oring' + str(i), rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()
