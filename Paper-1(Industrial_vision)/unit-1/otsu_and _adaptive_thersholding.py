import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r"D:\HCL\archive\hazelnut\test\hole\002.png", cv2.IMREAD_GRAYSCALE)

_, otsu_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 2)

# Gaussian adaptive threshold
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)

# -------- Plot Results --------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Original Grayscale")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Otsu Thresholding")
plt.imshow(otsu_mask, cmap='gray')
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Adaptive Mean Thresholding")
plt.imshow(adaptive_mean, cmap='gray')
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Adaptive Gaussian Thresholding")
plt.imshow(adaptive_gaussian, cmap='gray')
plt.axis("off")

plt.show()

