import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("006.png", cv2.IMREAD_GRAYSCALE)

# Laplacian Filtering (Edge Detection)
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)   # convert to uint8 for display

# Laplacian of Gaussian (LoG)
gaussian_blur = cv2.GaussianBlur(img, (5,5), sigmaX=1)

# Apply Laplacian on smoothed image

log = cv2.Laplacian(gaussian_blur, cv2.CV_64F, ksize=3)
log = cv2.convertScaleAbs(log)

# --- Display Results ---
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Laplacian Edges")
plt.imshow(laplacian, cmap='gray')
plt.axis("off")

plt.subplot(1,4,3)
plt.title("Gaussin blur")
plt.imshow(gaussian_blur, cmap='gray')
plt.axis("off")

plt.subplot(1,4,4)
plt.title("LoG (Laplacian of Gaussian)")
plt.imshow(log, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
