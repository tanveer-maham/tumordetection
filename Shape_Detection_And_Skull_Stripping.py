import numpy as np
import cv2

def SkullAndShape(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh != 0] = np.array((0, 0, 255))
    blended = cv2.addWeighted(img, 0.7, colormask, 0.1, 0)

    ret, markers = cv2.connectedComponents(thresh)

    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
    largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
    brain_mask = markers == largest_component
    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_out = img.copy()
    brain_out[closing == False] = (0, 0, 0)

    return brain_out