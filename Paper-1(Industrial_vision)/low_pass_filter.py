import cv2
import numpy as np

image = cv2.imread('006.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((10, 10), np.float32) / (10 * 10)

low_pass_filtered_image = cv2.filter2D(image, -1, kernel)
cv2.imshow('Original Image', image)
cv2.imshow('Low-Pass Filtered Image', low_pass_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
